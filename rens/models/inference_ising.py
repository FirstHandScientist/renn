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

def p2cbp_infer(ising, args):
    """
    Generalized BP, the parent2child algorithm.
    """
    model = ising
    model.generate_region_graph()
    
    gbp = parent2child_algo(graph=model.region_graph, n_iters=args.msg_iters)
    gbp.inference()
    gbp.read_beliefs()
    binary_marginals = torch.FloatTensor(len(model.binary_idx), 2, 2)
    for idx, pair in enumerate(model.binary_idx):
        if pair in gbp.graph.region_layers["R1"]:
            # for already gathered belief in gbp
            binary_marginals[idx] = torch.from_numpy(gbp.graph.nodes[pair]['belief'].values)
        else:
            # for belief not gathered in gbp
            pair_belief = 0
            parents_of_pair = gbp.graph.get_supernode_of(pair)
            assert len(parents_of_pair)==1
            for p_node in parents_of_pair:
                to_marginal_idx = tuple(sorted(set(p_node) - set(pair)))
                p_belief = gbp.graph.nodes[p_node]['belief'].copy()
                p_belief.marginalize([str(i) for i in to_marginal_idx], inplace=True)
                p_belief.normalize(inplace=True)
                pair_belief += p_belief.values

            binary_marginals[idx] = torch.from_numpy(pair_belief / len(parents_of_pair))
            
    unary_marginals = binary2unary_marginals(model.binary_idx, binary_marginals, model.n)
    
    return (None, unary_marginals, binary_marginals)


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
    log_Z_mf_energy = -ising.free_energy_mf(unary_marginals_mf)

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

    def forward(self):
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

        with torch.no_grad():
            unary_marginals_enc, binary_marginals_enc = self.encoder(self.ising.binary_idx)
        log_Z_enc = -self.ising.bethe_energy(unary_marginals_enc, binary_marginals_enc)  

        return (log_Z_enc, unary_marginals_enc, binary_marginals_enc)

class kikuchi_net_infer(torch.nn.Module):
    def __init__(self, ising, args):
        super(kikuchi_net_infer, self).__init__()
        self.model = ising
        self.model.generate_region_graph()
        self.encoder = ising_models.GeneralizedInferenceNetwork(args.n, args.state_dim, args.num_layers, mlp_out_dim=2**4)
        self.encoder.push2device(self.model.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.args = args

    def forward(self):
        unary_marginals_enc = torch.zeros(self.model.n ** 2).fill_(0.5).to(self.model.device)
        binary_marginals_enc = torch.zeros([len(self.model.binary_idx), 2, 2]).fill_(0.25).to(self.model.device)

        for i in range(self.args.enc_iters):
            self.optimizer.zero_grad()
            infer_beliefs, consist_error = self.encoder(self.model.region_graph)
            kikuchi_energy = self.encoder.kikuchi_energy(log_phis=self.model.log_phis,\
                                                         infer_beliefs=infer_beliefs, \
                                                         counts=self.model.region_graph.collect_region_count())
            loss = kikuchi_energy + self.args.agreement_pen * consist_error
            loss.backward()

            with torch.no_grad():
                # print(i,loss)
                unary_marginals_enc_new, binary_marginals_enc_new =\
                    self.encoder.read_marginals(binary_idx=self.model.binary_idx,\
                                                infer_beliefs=infer_beliefs, \
                                                graph=self.model.region_graph)

                delta_unary = l2(unary_marginals_enc_new, unary_marginals_enc) 
                delta_binary = l2(binary_marginals_enc_new[:, 1, 1], binary_marginals_enc[:, 1, 1])
                delta = delta_unary + delta_binary
                if delta < self.args.eps:
                    break

                unary_marginals_enc = unary_marginals_enc_new.detach()
                binary_marginals_enc = binary_marginals_enc_new.detach()

            self.optimizer.step()

        # compute the region based energy to estimated partition function

        self.optimizer.zero_grad()
        with torch.no_grad():
            infer_beliefs, consist_error = self.encoder(self.model.region_graph)

        kikuchi_energy = self.encoder.kikuchi_energy(log_phis=self.model.log_phis,\
                                                     infer_beliefs=infer_beliefs, \
                                                     counts=self.model.region_graph.collect_region_count())


        return (-kikuchi_energy, unary_marginals_enc, binary_marginals_enc)
