from mbr_nmt.mbr import mbr

def c2f_mbr(samples, utility1, topk, utility2=None, mc1=None, mc2=None, candidates=None,
            return_matrix=False, subsample_per_candidate=False):

   if utility2 is None:
       utility2 = utility1

   _, topk_candidates = mbr(samples, utility1, candidates=candidates, subsample_size=mc1,
                         return_matrix=False, subsample_per_candidate=subsample_per_candidate, return_topk=topk)
   pred_idx, pred, utility_matrix = mbr(samples, utility2, candidates=topk_candidates, subsample_size=mc2,
                                        return_matrix=True, subsample_per_candidate=subsample_per_candidate)

   if return_matrix:
       return pred_idx, pred, utility_matrix
   else:
       return pred_idx, pred
