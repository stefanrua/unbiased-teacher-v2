# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import sys
import os
def bcet(img): # from https://www.kaggle.com/code/kuuuuub/which-x-ray-preprocessing-performs-well
        #to_pil_image(img).save('before.png')
        img = np.array(img, dtype=float)
        Lmin = np.min(img)
        Lmax = np.max(img)
        Lmean = np.mean(img)
        LMssum = np.mean(img * img)

        # Output
        Gmin = 0
        Gmax = 255
        Gmean = 110

        bnum = Lmax * Lmax *(Gmean-Gmin) - LMssum*(Gmax-Gmin) + Lmin * Lmin *(Gmax-Gmean)
        bden = 2*(Lmax*(Gmean-Gmin)-Lmean*(Gmax-Gmin)+Lmin*(Gmax-Gmean))
        b = bnum/bden
        a = (Gmax-Gmin)/((Lmax-Lmin)*(Lmax+Lmin-2*b))
        c = Gmin - a*(Lmin-b) * (Lmin-b)
        y = a*(img-b) * (img-b) +c
        #y = (a*(img-b) * (img-b) +c) * -1 + 255
        y = np.clip(y, 0, 255)
        y = np.array(y, dtype=np.uint8)
        y = torch.Tensor(y)
        #to_pil_image(y).save('after.png')
        #sys.exit(0)
        return y

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            if os.getenv('BCET'):
                for i in batched_inputs:
                    i['image'] = bcet(i['image'])
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "unsup_data_train":  #

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
