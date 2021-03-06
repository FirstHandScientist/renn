import torch
from rens.models.GBP import parent2child_algo 
from rens.utils.utils import l2, get_scores, binary2unary_marginals
from rens.models import ising as ising_models
# inference methods ising model
def bp_infer(ising, args, solver):
    '''Do belief propagation with given solver'''
    msg_iters = args.msg_iters
    optmz_alpha = args.optmz_alpha

    messages = torch.zeros(ising.n**2, ising.n**2, 2).fill_(0.5).to(ising.device)
    unary_marginals_lbp, binary_marginals_lbp = ising.lbp_marginals(messages)
    
    if optmz_alpha:
        optimizer = torch.optim.Adam([ising.alpha_wgt], lr=0.005)

    for i in range(msg_iters):
        if solver is 'lbp':
            messages = ising.lbp_update(1, messages).detach()
            unary_marginals_lbp_new, binary_marginals_lbp_new = ising.lbp_marginals(messages)
        elif solver is 'dampbp':
            messages = ising.lbp_update(1, messages, args.damp).detach()
            unary_marginals_lbp_new, binary_marginals_lbp_new = ising.lbp_marginals(messages)
        elif solver is "trbp":
            messages = ising.trbp_update(1, messages, args.damp).detach()
            unary_marginals_lbp_new, binary_marginals_lbp_new = ising.trbp_marginals(messages)
        elif solver is 'alphabp':
            new_messages = ising.alphabp_update(1, messages)
            unary_marginals_lbp_new, binary_marginals_lbp_new = ising.alphabp_marginals(new_messages)
            if optmz_alpha:
                optimizer.zero_grad()
                loss = ising.free_energy_mf(unary_marginals_lbp_new)
                loss.backward()
                optimizer.step()
                for group in optimizer.param_groups:
                    group['params'][0].data.clamp_(-1, 1.5)


            messages = new_messages.detach()

        

        delta_unary = l2(unary_marginals_lbp_new, unary_marginals_lbp) 
        delta_binary = l2(binary_marginals_lbp_new[:, 1, 1], binary_marginals_lbp[:, 1, 1])
        delta = delta_unary + delta_binary
        if delta < args.eps:
            break

        unary_marginals_lbp = unary_marginals_lbp_new.detach()
        binary_marginals_lbp = binary_marginals_lbp_new.detach()

    log_Z_lbp = -ising.bethe_energy(unary_marginals_lbp, binary_marginals_lbp)

    return log_Z_lbp, unary_marginals_lbp, binary_marginals_lbp

class p2cbp_infer(torch.nn.Module):
    """
    Generalized BP, the parent2child algorithm.
    """
        
    def __init__(self, ising, args):
        super(p2cbp_infer, self).__init__()
        self.model = ising
        if self.model.region_graph == None:
            self.model.generate_region_graph()
        
        self.gbp = parent2child_algo(graph=self.model.region_graph, n_iters=args.msg_iters)

    def forward(self):
        self.gbp.inference()
        self.gbp.read_beliefs()
        kikuchi_energy = self.gbp.kikuchi_energy()
        binary_marginals = torch.FloatTensor(len(self.model.binary_idx), 2, 2)
        for idx, pair in enumerate(self.model.binary_idx):
            if pair in self.gbp.graph.region_layers["R1"]:
                # for already gathered belief in gbp
                binary_marginals[idx] = self.gbp.graph.nodes[pair]['belief'].values
            else:
                # for belief not gathered in gbp
                pair_belief = 0
                parents_of_pair = self.gbp.graph.get_supernode_of(pair)
                assert len(parents_of_pair)==1
                for p_node in parents_of_pair:
                    to_marginal_idx = tuple(sorted(set(p_node) - set(pair)))
                    p_belief = self.gbp.graph.nodes[p_node]['belief'].copy()
                    p_belief.marginalize([str(i) for i in to_marginal_idx], inplace=True)
                    p_belief.normalize(inplace=True)
                    pair_belief += p_belief.values

                binary_marginals[idx] = pair_belief / len(parents_of_pair)

        unary_marginals = binary2unary_marginals(self.model.binary_idx, binary_marginals, self.model.n)

        return (-kikuchi_energy, unary_marginals, binary_marginals)

    def neg_free_energy(self):
        self.model._init_disfactor()
        self.model.attach_region_factors(self.gbp.graph)
        kukuchi_energy = self.gbp.kikuchi_energy()
        
        energy = 0
        graph = self.gbp.graph
        for node in graph.nodes():
            graph.nodes[node]['belief'].values = torch.clamp(graph.nodes[node]['belief'].values.detach(), min=1e-8)
            energy += torch.sum(graph.nodes[node]['belief'].values * \
                                (graph.nodes[node]['belief'].values.log() - \
                                 graph.nodes[node]['log_phi'].values)) * \
                                graph.nodes[node]['weight']

        
        return (-energy, 0, 0)

        



def mean_field_infer(ising, args):
    """Run mean field algorithm. """ 

    unary_marginals_mf = torch.zeros(ising.n**2).fill_(0.5).to(ising.device)
    binary_marginals_mf = ising.mf_binary_marginals(unary_marginals_mf)
    
    for i in range(args.msg_iters):
        unary_marginals_mf_new = ising.mf_update(1, unary_marginals_mf)
        binary_marginals_mf_new = ising.mf_binary_marginals(unary_marginals_mf_new)
        delta_unary = l2(unary_marginals_mf_new, unary_marginals_mf) 
        delta_binary = l2(binary_marginals_mf_new[:, 1, 1], binary_marginals_mf[:, 1, 1])
        delta = delta_unary + delta_binary
        if delta < args.eps:
            break
        
        unary_marginals_mf = unary_marginals_mf_new.detach()
        binary_marginals_mf = binary_marginals_mf_new.detach()

    log_Z_mf = -ising.bethe_energy(unary_marginals_mf, binary_marginals_mf)
    log_Z_mf_energy = -ising.free_energy_mf(unary_marginals_mf.detach())

    return (log_Z_mf_energy, unary_marginals_mf, binary_marginals_mf)

class bethe_net_infer(torch.nn.Module):
    def __init__(self, ising, args):
        # inference network
        super(bethe_net_infer, self).__init__()
        self.ising = ising
        self.device = args.device
        self.encoder = ising_models.TransformerInferenceNetwork(args.n, args.state_dim, args.num_layers)
        self.encoder.to(self.device)
        self.encoder.device = self.device
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.args = args

    def forward(self, learn_model=False):
        unary_marginals_enc = torch.zeros(self.ising.n ** 2).fill_(0.5).to(self.device)
        binary_marginals_enc = torch.zeros([len(self.ising.binary_idx), 2, 2]).fill_(0.25).to(self.device)

        for i in range(self.args.enc_iters):
            self.optimizer.zero_grad()
            unary_marginals_enc_new, binary_marginals_enc_new = self.encoder(self.ising.binary_idx)
            bethe_enc = self.ising.bethe_energy(unary_marginals_enc_new, binary_marginals_enc_new)
            agreement_loss = self.encoder.agreement_penalty(self.ising.binary_idx, unary_marginals_enc_new,
                                                       binary_marginals_enc_new)
            (bethe_enc + self.args.agreement_pen*agreement_loss).backward()

            self.optimizer.step()
            delta_unary = l2(unary_marginals_enc_new, unary_marginals_enc) 
            delta_binary = l2(binary_marginals_enc_new[:, 1, 1], binary_marginals_enc[:, 1, 1])
            delta = delta_unary + delta_binary
            if delta < self.args.eps:
                break

            unary_marginals_enc = unary_marginals_enc_new.detach()
            binary_marginals_enc = binary_marginals_enc_new.detach()


        self.unary_marginals_enc = unary_marginals_enc.detach()
        self.binary_marginals_enc = binary_marginals_enc.detach()
        
        if not learn_model:
            log_Z_enc = -self.ising.bethe_energy(unary_marginals_enc, binary_marginals_enc)
            return (log_Z_enc, unary_marginals_enc, binary_marginals_enc)
        else:
            bethe_enc = self.ising.bethe_energy(unary_marginals_enc, binary_marginals_enc)
            agreement_loss = self.encoder.agreement_penalty(self.ising.binary_idx, unary_marginals_enc, binary_marginals_enc)

            return (-bethe_enc, - agreement_loss, len(self.ising.binary_idx))
    def neg_free_energy(self):
        bethe_enc = self.ising.bethe_energy(self.unary_marginals_enc, self.binary_marginals_enc)
        return -bethe_enc, 0, 0


class kikuchi_net_infer(torch.nn.Module):
    def __init__(self, ising, args):
        super(kikuchi_net_infer, self).__init__()
        self.model = ising
        if self.model.region_graph == None:
            self.model.generate_region_graph()
            
        self.encoder = ising_models.GeneralizedInferenceNetwork(args.n, args.state_dim, args.num_layers, mlp_out_dim=2**self.model.r0_elmt_size)
        self.encoder.push2device(self.model.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.args = args

    def forward(self, learn_model=False):
        unary_marginals_enc = torch.zeros(self.model.n ** 2).fill_(0.5).to(self.model.device)
        binary_marginals_enc = torch.zeros([len(self.model.binary_idx), 2, 2]).fill_(0.25).to(self.model.device)

        for i in range(self.args.enc_iters):
            self.optimizer.zero_grad()
            kikuchi_energy, consist_error = self.encoder(self.model.region_graph)
            
            loss = kikuchi_energy + self.args.agreement_pen * consist_error
            loss.backward()
            
            with torch.no_grad():
                # print(i,loss)
                unary_marginals_enc_new, binary_marginals_enc_new =\
                    self.encoder.read_marginals(binary_idx=self.model.binary_idx,\
                                                num_nodes=self.model.n, \
                                                graph=self.model.region_graph)

                delta_unary = l2(unary_marginals_enc_new, unary_marginals_enc) 
                delta_binary = l2(binary_marginals_enc_new[:, 1, 1], binary_marginals_enc[:, 1, 1])
                delta = delta_unary + delta_binary
                if delta < self.args.eps and i > 10:
                    break

                unary_marginals_enc = unary_marginals_enc_new.detach()
                binary_marginals_enc = binary_marginals_enc_new.detach()

            self.optimizer.step()

        # compute the region based energy to estimated partition function
        kikuchi_energy, consist_error = self.encoder(self.model.region_graph)
        if not learn_model:
            return (-kikuchi_energy, unary_marginals_enc, binary_marginals_enc)
        else:
            match_node_num = int(self.model.region_graph.number_of_nodes()) - \
                len(self.model.region_graph.region_layers['R0'])
            return (-kikuchi_energy, -consist_error, match_node_num)

    def neg_free_energy(self):
        """Compute the Kikuchi free energy"""
        # kikuchi_energy, consist_error = self.encoder(self.model.region_graph)
        graph = self.model.region_graph
        energy = 0
        belief_name = 'net_belief'
        self.model._init_disfactor()
        self.model.attach_region_factors(graph)
        for node in graph.nodes():
            graph.nodes[node][belief_name].values = torch.clamp(graph.nodes[node][belief_name].values.detach(), min=1e-8)
            energy += torch.sum(graph.nodes[node][belief_name].values * \
                       (graph.nodes[node][belief_name].values.log() - \
                        graph.nodes[node]['log_phi'].values)) * \
                        graph.nodes[node]['weight']

        # energy
        
        match_node_num = int(self.model.region_graph.number_of_nodes()) - \
            len(self.model.region_graph.region_layers['R0'])
        return (-energy, 0, match_node_num)


        
