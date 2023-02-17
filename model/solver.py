import torch
import torchvision


def box_nms(prob, size=4, iou=0.1, min_prob=0.015, keep_top_k=-1):
    """
    :param prob: probability, torch.tensor, must be [1,H,W]
    :param size: box size for 2d nms
    :param iou:
    :param min_prob:
    :param keep_top_k:
    :return:
    """
    assert(prob.shape[0]==1 and len(prob.shape)==3)
    prob = prob.squeeze(dim=0)

    pts = torch.stack(torch.where(prob>=min_prob)).t()
    boxes = torch.cat((pts-size/2.0, pts+size/2.0),dim=1).to(torch.float32)
    scores = prob[pts[:,0],pts[:,1]]
    indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou)
    pts = pts[indices,:]
    scores = scores[indices]
    if keep_top_k>0:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores,k)
        pts = pts[indices,:]
    nms_prob = torch.zeros_like(prob)
    nms_prob[pts[:,0],pts[:,1]] = scores

    return nms_prob.unsqueeze(dim=0)
