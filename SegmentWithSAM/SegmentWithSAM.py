import glob
import os
import pickle
import numpy as np
import qt
import vtk
import shutil
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
)
from slicer.util import VTKObservationMixin
import SampleData
from PIL import Image

#
# SegmentWithSAM
#

class SegmentWithSAM(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SegmentWithSAM"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Zafer Yildiz (Mazurowski Lab, Duke University)"]
        self.parent.helpText = """
The SegmentWithSAM module aims to assist its users in segmenting medical data by integrating
the <a href="https://github.com/facebookresearch/segment-anything">Segment Anything Model (SAM)</a>
developed by Meta.<br>
<br>
See more information in <a href="https://github.com/mazurowski-lab/SlicerSegmentWithSAM">module documentation</a>.
"""
        self.parent.acknowledgementText = """
This file was originally developed by Zafer Yildiz (Mazurowski Lab, Duke University).
"""


#
# SegmentWithSAMWidget
#


class SegmentWithSAMWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        global sam_model_registry
        global SamPredictor
        global torch
        global cv2
        global hydra
        global tqdm
        global build_sam2
        global SAM2ImagePredictor
        global build_sam2_video_predictor
        global sam2_setup
        global setuptools
        global ninja
        global plt
        

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.slicesFolder = self.resourcePath("UI") + "/../../../slices"
        self.featuresFolder = self.resourcePath("UI") + "/../../../features"
        self.framesFolder = self.resourcePath("UI") + "/../../../frames"

        self.modelVersion = "vit_h"
        self.checkpointName = "sam_vit_h_4b8939.pth"
        self.checkpointFolder = self.resourcePath("UI") + "/../../../model_checkpoints/"
        self.modelCheckpoint = self.checkpointFolder + self.checkpointName
        self.masks = None
        self.mask_threshold = 0

        vtk.vtkObject.GlobalWarningDisplayOff()

        if not os.path.exists(self.checkpointFolder):
            os.makedirs(self.checkpointFolder)

        if not os.path.exists(self.checkpointFolder + "sam_vit_h_4b8939.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM (ViT-H) checkpoint (2.38 GB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam_vit_h_4b8939.pth")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam_vit_l_0b3195.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM (ViT-L) checkpoint (1.16 GB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "sam_vit_l_0b3195.pth", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam_vit_l_0b3195.pth")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam_vit_b_01ec64.pth"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM (ViT-B) checkpoint (357 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "sam_vit_b_01ec64.pth", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam_vit_b_01ec64.pth")
                    slicer.progressWindow.close()
        
        if not os.path.exists(self.checkpointFolder + "sam2_hiera_tiny.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Tiny) checkpoint (148 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt", "sam2_hiera_tiny.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_tiny.pt")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam2_hiera_small.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Small) checkpoint (175 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt", "sam2_hiera_small.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_small.pt")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam2_hiera_base_plus.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Base Plus) checkpoint (308 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", "sam2_hiera_base_plus.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_base_plus.pt")
                    slicer.progressWindow.close()

        if not os.path.exists(self.checkpointFolder + "sam2_hiera_large.pt"):
            if slicer.util.confirmOkCancelDisplay(
                "Would you like to use SAM-2 (Base Plus) checkpoint (856 MB)? Click OK to install it now!"
            ):
                
                slicer.progressWindow = slicer.util.createProgressDialog()
                self.sampleDataLogic = SampleData.SampleDataLogic()
                self.sampleDataLogic.logMessage = self.reportProgress

                checksum = "SHA256:7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b"    
                downloadedFilePath = self.sampleDataLogic.downloadFileIntoCache("https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt", "sam2_hiera_large.pt", checksum)
                
                if self.sampleDataLogic.downloadPercent and self.sampleDataLogic.downloadPercent == 100:
                    shutil.copyfile(downloadedFilePath, self.checkpointFolder + "sam2_hiera_large.pt")
                    slicer.progressWindow.close()

        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            extensionName = 'PyTorch'
            em = slicer.app.extensionsManagerModel()
            em.interactive = False  # prevent display of popups
            restart = True
            if not em.installExtensionFromServer(extensionName, restart):
                raise ValueError(f"Failed to install {extensionName} extension")

        minimumTorchVersion = "2.0.0"
        minimumTorchVisionVersion = "0.15.0"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()

        if not torchLogic.torchInstalled():
            slicer.util.delayDisplay("PyTorch Python package is required. Installing... (it may take several minutes)")
            torch = torchLogic.installTorch(
                askConfirmation=True,
                forceComputationBackend="cu117",
                torchVersionRequirement=f">={minimumTorchVersion}",
                torchvisionVersionRequirement=f">={minimumTorchVisionVersion}",
            )
            if torch is None:
                raise ValueError("You need to install PyTorch to use SegmentWithSAM!")
        else:
            # torch is installed, check version
            from packaging import version

            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(
                    f"PyTorch version {torchLogic.torch.__version__} is not compatible with this module."
                    ' Minimum required version is {minimumTorchVersion}. You can use "PyTorch Util" module'
                    " to install PyTorch with version requirement set to: >={minimumTorchVersion}"
                )

        import torch

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'segment-anything' is missing. Click OK to install it now!"
            ):
                slicer.util.pip_install("https://github.com/facebookresearch/segment-anything/archive/6fdee8f2727f4506cfbbe553e23b895e27956588.zip") 
        try: 
            from segment_anything import sam_model_registry, SamPredictor
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'segment-anything' package. Please try again to install!")
        
        try:
            import hydra
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'hydra' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("hydra-core")
        import hydra

        try:
            import ninja
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'ninja' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("ninja")
        import ninja

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'matplotlib' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("matplotlib")
        import matplotlib.pyplot as plt

        try:
            import tqdm
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'tqdm' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("tqdm")
        import tqdm

        try:
            import setuptools
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'setuptools' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("setuptools==69.5.1")
        import setuptools
        '''print(os.listdir(self.resourcePath("UI")))
        print(os.listdir(self.resourcePath("UI") + "/../../../") )
        os.chdir(self.resourcePath("UI") + "/../../../")
        os.system("python setup.py build_ext --inplace" )'''
        try: 
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.build_sam import build_sam2_video_predictor
            #from setup import setup as sam2setup
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'sam-2' package. Please try again to install!")
        
        #setup(cmdclass="build_ext")

        try:
            import cv2
        except ModuleNotFoundError:
            if slicer.util.confirmOkCancelDisplay(
                "'open-cv' is missing. Click OK to install it now!"
            ): 
                slicer.util.pip_install("opencv-python")

        try: 
            import cv2
        except ModuleNotFoundError:
            raise RuntimeError("There is a problem about the installation of 'open-cv' package. Please try again to install!")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Working on", self.device)

        self.currentlySegmenting = False
        self.featuresAreExtracted = False

    def createFrames(self):
        if not os.path.exists(self.framesFolder):
            os.makedirs(self.framesFolder)

        oldSliceFiles = glob.glob(self.framesFolder + "/*")
        for filename in oldSliceFiles:
            os.remove(filename)
        self.initializeVariables()
        self.initializeSegmentationProcess()

        for sliceIndex in range(0, self.nofSlices):
            sliceImage = self.getSliceBasedOnSliceAccessorDimension(sliceIndex)
            plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(sliceIndex))) + str(sliceIndex) + ".jpeg", sliceImage, cmap="gray")


    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def propagation(self, toLeft):

        frame_names = [
            p for p in os.listdir(self.framesFolder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        inference_state = self.videoPredictor.init_state(video_path=self.framesFolder)
        self.videoPredictor.reset_state(inference_state)

        sliceIndicesToPromptPointCoordinations, sliceIndicesToPromptPointLabels = self.getAllPromptPointsAndLabels()

        ann_frame_idx =  self.getIndexOfCurrentSlice()  # the frame index we interact with
        ann_obj_id = self._parameterNode.GetParameter("SAMCurrentSegment") 
        # give a unique id to each object we interact with (it can be any integers)
        
        _, out_obj_ids, out_mask_logits = self.videoPredictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=np.array(sliceIndicesToPromptPointCoordinations[ann_frame_idx], dtype=np.float32),
            labels=np.array(sliceIndicesToPromptPointLabels[ann_frame_idx], np.int32),
        )
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.videoPredictor.propagate_in_video(inference_state, reverse=toLeft):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        plt.close("all")
        
        if toLeft:
            for out_frame_idx in range(0, self.getIndexOfCurrentSlice() + 1):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    orig_map=plt.cm.get_cmap('binary') 
                    # reversing the original colormap using reversed() function 
                    reversed_binary = orig_map.reversed() 
                    plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(out_frame_idx))) + str(out_frame_idx) + "_mask.jpeg", out_mask[0], cmap=reversed_binary)
            
            self.updateSlicesWithSegmentationMasks(0, self.getIndexOfCurrentSlice())
        else:
            for out_frame_idx in range(self.getIndexOfCurrentSlice(), self.nofSlices):
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    orig_map=plt.cm.get_cmap('binary') 
                    # reversing the original colormap using reversed() function 
                    reversed_binary = orig_map.reversed() 
                    plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(out_frame_idx))) + str(out_frame_idx) + "_mask.jpeg", out_mask[0], cmap=reversed_binary)

            self.updateSlicesWithSegmentationMasks(self.getIndexOfCurrentSlice(), self.nofSlices - 1)

    def getAllPromptPointsAndLabels(self):
        sliceIndicesToPromptPointCoordinations = {}
        sliceIndicesToPromptPointLabels = {}

        nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()
        for i in range(nofPositivePromptPoints):
            if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                pointRAS = [0, 0, 0]
                self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                pointIJK = [0, 0, 0, 1]
                self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                if self.sliceAccessorDimension == 2:
                    if pointIJK[0] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[0]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[0]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[0]].append([pointIJK[1], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[0]].append(1)
                elif self.sliceAccessorDimension == 1:
                    if pointIJK[1] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[1]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[1]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[1]].append([pointIJK[0], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[1]].append(1)

                elif self.sliceAccessorDimension == 0:
                    if pointIJK[2] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[2]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[2]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[2]].append([pointIJK[0], pointIJK[1]])
                    sliceIndicesToPromptPointLabels[pointIJK[2]].append(1)

        nofNegativePromptPoints = self.negativePromptPointsNode.GetNumberOfControlPoints()
        for i in range(nofNegativePromptPoints):
            if self.negativePromptPointsNode.GetNthControlPointVisibility(i):
                pointRAS = [0, 0, 0]
                self.negativePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                pointIJK = [0, 0, 0, 1]
                self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                if self.sliceAccessorDimension == 2:
                    if pointIJK[0] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[0]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[0]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[0]].append([pointIJK[1], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[0]].append(0)
                elif self.sliceAccessorDimension == 1:
                    if pointIJK[1] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[1]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[1]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[1]].append([pointIJK[0], pointIJK[2]])
                    sliceIndicesToPromptPointLabels[pointIJK[1]].append(0)
                elif self.sliceAccessorDimension == 0:
                    if pointIJK[2] not in sliceIndicesToPromptPointCoordinations.keys():
                        sliceIndicesToPromptPointCoordinations[pointIJK[2]] = []
                        sliceIndicesToPromptPointLabels[pointIJK[2]] = []
                    sliceIndicesToPromptPointCoordinations[pointIJK[2]].append([pointIJK[0], pointIJK[1]])
                    sliceIndicesToPromptPointLabels[pointIJK[2]].append(0)

        return sliceIndicesToPromptPointCoordinations, sliceIndicesToPromptPointLabels


    def propagateThroughAllSlices(self):
        with slicer.util.MessageDialog("Propagating through all slices..."):
            with slicer.util.WaitCursor():
                self.createFrames()

                sliceIndicesToPromptPointCoordinations, sliceIndicesToPromptPointLabels = self.getAllPromptPointsAndLabels()
                
                frame_names = [
                    p for p in os.listdir(self.framesFolder)
                    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
                ]
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
                inference_state = self.videoPredictor.init_state(video_path=self.framesFolder)

                for ann_frame_idx in sliceIndicesToPromptPointCoordinations.keys():
                    _, out_obj_ids, out_mask_logits = self.videoPredictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=self._parameterNode.GetParameter("SAMCurrentSegment"),
                        points=np.array(sliceIndicesToPromptPointCoordinations[ann_frame_idx], dtype=np.float32),
                        labels=np.array(sliceIndicesToPromptPointLabels[ann_frame_idx], np.int32),
                    )

                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in self.videoPredictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                for out_frame_idx, out_obj_ids, out_mask_logits in self.videoPredictor.propagate_in_video(inference_state, reverse=True):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                plt.close("all")
                # render the segmentation results every few frames
                for out_frame_idx in range(len(frame_names)):
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        orig_map=plt.cm.get_cmap('binary') 
                        # reversing the original colormap using reversed() function 
                        reversed_binary = orig_map.reversed() 
                        plt.imsave(self.framesFolder + "/" + "0" * (5 - len(str(out_frame_idx))) + str(out_frame_idx) + "_mask.jpeg", out_mask[0], cmap=reversed_binary)

                self.updateSlicesWithSegmentationMasks(0, self.nofSlices - 1)


    def updateSlicesWithSegmentationMasks(self, start, end):
        currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")

        if currentSegment not in self.segmentIdToSegmentationMask:
            self.segmentIdToSegmentationMask[currentSegment] = np.zeros(self.volumeShape)

        for currentSliceIndex in range(start, end + 1):
            mask = Image.open(self.framesFolder + "/" +  "0" * (5 - len(str(currentSliceIndex))) + str(currentSliceIndex) + "_mask.jpeg")  
            mask = mask.convert('1')
            mask = np.array(mask)

            if self.sliceAccessorDimension == 2:
                self.segmentIdToSegmentationMask[currentSegment][:, :, currentSliceIndex] = mask
            elif self.sliceAccessorDimension == 1:
                self.segmentIdToSegmentationMask[currentSegment][:, currentSliceIndex, :] = mask
            else:
                self.segmentIdToSegmentationMask[currentSegment][currentSliceIndex, :, :] = mask

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            self.segmentIdToSegmentationMask[currentSegment],
            self._parameterNode.GetNodeReference("SAMSegmentationNode"),
            self._parameterNode.GetParameter("SAMCurrentSegment"),
            self._parameterNode.GetNodeReference("InputVolume"),
        )

    def propagateToLeft(self):
        with slicer.util.MessageDialog("Propagating to left..."):
            with slicer.util.WaitCursor():
                self.createFrames()
                self.propagation(toLeft=True)
                self.positivePromptPointsNode.RemoveAllControlPoints()
                self.negativePromptPointsNode.RemoveAllControlPoints()


    def propagateToRight(self):
        with slicer.util.MessageDialog("Propagating to right..."):
            with slicer.util.WaitCursor():
                self.createFrames()
                self.propagation(toLeft=False)

                self.positivePromptPointsNode.RemoveAllControlPoints()
                self.negativePromptPointsNode.RemoveAllControlPoints()

    def changeModel(self, modelName):
        self.modelName = modelName

        if "SAM-2" in modelName:
            if self.device.type == "cuda":
                # use bfloat16 for the entire notebook
                #torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

        if modelName == "SAM-2 Large":
            sam2_checkpoint = self.checkpointFolder + "sam2_hiera_large.pt"
            model_cfg = "sam2_hiera_l.yaml"
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.videoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            self.sam = SAM2ImagePredictor(sam2_model)
        
        elif modelName == "SAM-2 Base Plus":
            sam2_checkpoint = self.checkpointFolder + "sam2_hiera_base_plus.pt"
            model_cfg = "sam2_hiera_b+.yaml"
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.videoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            self.sam = SAM2ImagePredictor(sam2_model)
        
        elif modelName == "SAM-2 Small":
            sam2_checkpoint = self.checkpointFolder + "sam2_hiera_small.pt"
            model_cfg = "sam2_hiera_s.yaml"
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.videoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            self.sam = SAM2ImagePredictor(sam2_model)

        elif modelName == "SAM-2 Tiny":
            sam2_checkpoint = self.checkpointFolder + "sam2_hiera_tiny.pt"
            model_cfg = "sam2_hiera_t.yaml"
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.videoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            self.sam = SAM2ImagePredictor(sam2_model)
        
        elif modelName == "SAM (ViT-B)":
            self.modelVersion = "vit_b"
            self.checkpointName = "sam_vit_b_01ec64.pth"
            self.modelCheckpoint = self.checkpointFolder + self.checkpointName
            model = sam_model_registry[self.modelVersion](checkpoint=self.modelCheckpoint)
            model.to(device=self.device)
            self.sam = SamPredictor(model)
        
        elif modelName == "SAM (ViT-L)":
            self.modelVersion = "vit_l"
            self.checkpointName = "sam_vit_l_0b3195.pth"
            self.modelCheckpoint = self.checkpointFolder + self.checkpointName
            model = sam_model_registry[self.modelVersion](checkpoint=self.modelCheckpoint)
            model.to(device=self.device)
            self.sam = SamPredictor(model)
        
        elif modelName == "SAM (ViT-H)":
            self.modelVersion = "vit_h"
            self.checkpointName = "sam_vit_h_4b8939.pth"
            self.modelCheckpoint = self.checkpointFolder + self.checkpointName
            model = sam_model_registry[self.modelVersion](checkpoint=self.modelCheckpoint)
            model.to(device=self.device)
            self.sam = SamPredictor(model)

        self.currentlySegmenting = False
        self.featuresAreExtracted = False

    def reportProgress(self, msg, level=None):
        # Print progress in the console
        print("Loading... {0}%".format(self.sampleDataLogic.downloadPercent))
        # Abort download if cancel is clicked in progress bar
        if slicer.progressWindow.wasCanceled:
            raise Exception("Download aborted")
        # Update progress window
        slicer.progressWindow.show()
        slicer.progressWindow.activateWindow()
        slicer.progressWindow.setValue(int(self.sampleDataLogic.downloadPercent))
        slicer.progressWindow.setLabelText("Downloading SAM checkpoint...")
        # Process events to allow screen to refresh
        slicer.app.processEvents()

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentWithSAM.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = SegmentWithSAMLogic()

        # Connections
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.ui.positivePrompts.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.positivePrompts.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.negativePrompts.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.negativePrompts.markupsPlaceWidget().setPlaceModePersistency(True)

        # Buttons
        self.ui.goToSegmentEditorButton.connect("clicked(bool)", self.onGoToSegmentEditor)
        self.ui.goToMarkupsButton.connect("clicked(bool)", self.onGoToMarkups)
        self.ui.propagateToLeft.connect("clicked(bool)", self.propagateToLeft)
        self.ui.propagateToRight.connect("clicked(bool)", self.propagateToRight)
        self.ui.propagateThroughAllSlices.connect('clicked(bool)', self.propagateThroughAllSlices)
        self.ui.segmentButton.connect("clicked(bool)", self.onStartSegmentation)
        self.ui.stopSegmentButton.connect("clicked(bool)", self.onStopSegmentButton)
        self.ui.segmentationDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.maskDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.modelDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)

        self.segmentIdToSegmentationMask = {}
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        #         
        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.GetNodeReferenceID("positivePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "positive")
            newPromptPointNode.GetDisplayNode().SetSelectedColor(0, 1, 0)
            self._parameterNode.SetNodeReferenceID("positivePromptPointsNode", newPromptPointNode.GetID())

        if not self._parameterNode.GetNodeReferenceID("negativePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "negative")
            newPromptPointNode.GetDisplayNode().SetSelectedColor(1, 0, 0)
            self._parameterNode.SetNodeReferenceID("negativePromptPointsNode", newPromptPointNode.GetID())

        self.ui.positivePrompts.setCurrentNode(self._parameterNode.GetNodeReference("positivePromptPointsNode"))
        self.ui.negativePrompts.setCurrentNode(self._parameterNode.GetNodeReference("negativePromptPointsNode"))

        if not self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):

            self.samSegmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'SAM Segmentation')
            self.samSegmentationNode.CreateDefaultDisplayNodes() 
            self.firstSegmentId = self.samSegmentationNode.GetSegmentation().AddEmptySegment()
            
            self._parameterNode.SetNodeReferenceID("SAMSegmentationNode", self.samSegmentationNode.GetID())
            self._parameterNode.SetParameter("SAMCurrentSegment", self.firstSegmentId)
            self._parameterNode.SetParameter("SAMCurrentMask", "Mask-1")
            self._parameterNode.SetParameter("SAMCurrentModel", "SAM")

            self.ui.segmentationDropDown.addItem(self.samSegmentationNode.GetSegmentation().GetNthSegment(0).GetName())
            for i in range(3):
                self.ui.maskDropDown.addItem("Mask-" + str(i+1))

            self.ui.modelDropDown.addItem("SAM-2 Tiny")
            self.ui.modelDropDown.addItem("SAM-2 Small")
            self.ui.modelDropDown.addItem("SAM-2 Base Plus")
            self.ui.modelDropDown.addItem("SAM-2 Large")
            
            self.ui.modelDropDown.addItem("SAM (ViT-B)")
            self.ui.modelDropDown.addItem("SAM (ViT-L)")
            self.ui.modelDropDown.addItem("SAM (ViT-H)")
            
    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None and self.hasObserver(
            self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
        ):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        if not slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode"):
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        if self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")

            self.ui.segmentationDropDown.clear()
            for i in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
                segmentName = segmentationNode.GetSegmentation().GetNthSegment(i).GetName()
                self.ui.segmentationDropDown.addItem(segmentName)
 
            if self._parameterNode.GetParameter("SAMCurrentSegment"):
                self.ui.segmentationDropDown.setCurrentText(segmentationNode.GetSegmentation().GetSegment(self._parameterNode.GetParameter("SAMCurrentSegment")).GetName())
            

            if self._parameterNode.GetParameter("SAMCurrentMask"):
                self.ui.maskDropDown.setCurrentText(self._parameterNode.GetParameter("SAMCurrentMask"))
            
            if self._parameterNode.GetParameter("SAMCurrentModel"):
                self.ui.modelDropDown.setCurrentText(self._parameterNode.GetParameter("SAMCurrentModel"))

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        if self._parameterNode.GetParameter("SAMCurrentModel") != self.ui.modelDropDown.currentText:
            print(self._parameterNode.GetParameter("SAMCurrentModel"), "=>", self.ui.modelDropDown.currentText)
            self.changeModel(self.ui.modelDropDown.currentText)
            self._parameterNode.SetParameter("SAMCurrentModel", self.ui.modelDropDown.currentText)

        if not self._parameterNode.GetNodeReference("SAMSegmentationNode") or not hasattr(self, 'volumeShape'):
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode").GetSegmentation()
        self._parameterNode.SetParameter("SAMCurrentSegment", segmentationNode.GetSegmentIdBySegmentName(self.ui.segmentationDropDown.currentText))
        if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
            self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)

        self._parameterNode.SetParameter("SAMCurrentMask", self.ui.maskDropDown.currentText)
        self._parameterNode.SetParameter("SAMCurrentModel", self.ui.modelDropDown.currentText)

        self._parameterNode.EndModify(wasModified)

    def initializeVariables(self):
        
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
                self.volume = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("InputVolume"))
                self.volumeShape = self.volume.shape
                if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)
                self.sliceAccessorDimension = self.getSliceAccessorDimension()
                sampleInputImage = None

                if self.sliceAccessorDimension == 0:
                    sampleInputImage = self.volume[0,:,:]
                    self.nofSlices = self.volume.shape[0]
                elif self.sliceAccessorDimension == 1:
                    sampleInputImage = self.volume[:,0,:]
                    self.nofSlices = self.volume.shape[1]
                else:
                    sampleInputImage = self.volume[:,:,0]
                    self.nofSlices = self.volume.shape[2]

                self.imageShape = sampleInputImage.shape
            else:
                slicer.util.warningDisplay("You need to add data first to start segmentation!")
                return False
        
        return True
            
    def createSlices(self):
        if not os.path.exists(self.slicesFolder):
            os.makedirs(self.slicesFolder)

        oldSliceFiles = glob.glob(self.slicesFolder + "/*")
        for filename in oldSliceFiles:
            os.remove(filename)

        for sliceIndex in range(self.nofSlices):
            sliceImage = self.getSliceBasedOnSliceAccessorDimension(sliceIndex)
            np.save(self.slicesFolder + "/" + f"slice_{sliceIndex}", sliceImage)

    def getSliceBasedOnSliceAccessorDimension(self, sliceIndex):
        if self.sliceAccessorDimension == 0:
            return self.volume[sliceIndex, :, :]
        elif self.sliceAccessorDimension == 1:
            return self.volume[:, sliceIndex, :]
        else:
            return self.volume[:, :, sliceIndex]

    def createFeatures(self):
        if not os.path.exists(self.featuresFolder):
            os.makedirs(self.featuresFolder)

        oldFeatureFiles = glob.glob(self.featuresFolder + "/*")
        for filename in oldFeatureFiles:
            os.remove(filename)

        for filename in os.listdir(self.slicesFolder):
            #if filename in ('slice_68.npy', 'slice_69.npy', 'slice_70.npy'):
            image = np.load(self.slicesFolder + "/" + filename)
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.sam.set_image(image)

            with open(self.featuresFolder + "/" + os.path.splitext(filename)[0] + "_features.pkl", "wb") as f:
                if "SAM-2" in self.modelName:
                    pickle.dump(self.sam._features, f)
                else:
                    pickle.dump(self.sam.features, f)

    def onStartSegmentation(self):
        if not self.initializeVariables():
            return

        currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
        currentSliceIndex = self.getIndexOfCurrentSlice()
        previouslyProducedMask = None

        if self.sliceAccessorDimension == 2:
            previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][:, :, currentSliceIndex]
        elif self.sliceAccessorDimension == 1:
            previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][:, currentSliceIndex, :]
        else:
            previouslyProducedMask = self.segmentIdToSegmentationMask[currentSegment][currentSliceIndex, :, :]

        if np.any(previouslyProducedMask):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
            currentLabel = segmentationNode.GetSegmentation().GetSegment(currentSegment).GetName()

            confirmed = slicer.util.confirmOkCancelDisplay(
                f"Are you sure you want to re-annotate {currentLabel} for the current slice? All of your previous annotation for {currentLabel} in the current slice will be removed!",
                windowTitle="Warning",
            )
            if not confirmed:
                return

        if not self.featuresAreExtracted:
            self.extractFeatures()
            self.featuresAreExtracted = True

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)

        self.currentlySegmenting = True
        self.initializeSegmentationProcess()
        self.positivePromptPointsNode.RemoveAllControlPoints()
        self.negativePromptPointsNode.RemoveAllControlPoints()
        self.collectPromptInputsAndPredictSegmentationMask()
        self.updateSegmentationScene()

    def getSliceAccessorDimension(self):
        npArray = np.zeros((3, 3))
        self._parameterNode.GetNodeReference("InputVolume").GetIJKToRASDirections(npArray)
        npArray = np.transpose(npArray)[0]
        maxIndex = 0
        maxValue = np.abs(npArray[0])

        for index in range(len(npArray)):
            if np.abs(npArray[index]) > maxValue:
                maxValue = np.abs(npArray[index])
                maxIndex = index

        return maxIndex

    def onStopSegmentButton(self):
        self.currentlySegmenting = False
        self.masks = None
        self.init_masks = None

        self.positivePromptPointsNode.RemoveAllControlPoints()
        self.negativePromptPointsNode.RemoveAllControlPoints()

        roiList = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiList:
            slicer.mrmlScene.RemoveNode(roiNode)

    def onGoToSegmentEditor(self):
        slicer.util.selectModule("SegmentEditor")

    def onGoToMarkups(self):
        slicer.util.selectModule("Markups")

    def getIndexOfCurrentSlice(self):
        redView = slicer.app.layoutManager().sliceWidget("Red")
        redViewLogic = redView.sliceLogic()
        return redViewLogic.GetSliceIndexFromOffset(redViewLogic.GetSliceOffset()) - 1

    def updateSegmentationScene(self):
        if self.currentlySegmenting:
            currentSegment = self._parameterNode.GetParameter("SAMCurrentSegment")
            currentSliceIndex = self.getIndexOfCurrentSlice()

            if currentSegment not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[currentSegment] = np.zeros(self.volumeShape)
            if self.sliceAccessorDimension == 2:
                self.segmentIdToSegmentationMask[currentSegment][:, :, currentSliceIndex] = self.producedMask
            elif self.sliceAccessorDimension == 1:
                self.segmentIdToSegmentationMask[currentSegment][:, currentSliceIndex, :] = self.producedMask
            else:
                self.segmentIdToSegmentationMask[currentSegment][currentSliceIndex, :, :] = self.producedMask

            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[currentSegment],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self._parameterNode.GetParameter("SAMCurrentSegment"),
                self._parameterNode.GetNodeReference("InputVolume"),
            )

        qt.QTimer.singleShot(100, self.updateSegmentationScene)

    def initializeSegmentationProcess(self):
        self.positivePromptPointsNode = self._parameterNode.GetNodeReference("positivePromptPointsNode")
        self.negativePromptPointsNode = self._parameterNode.GetNodeReference("negativePromptPointsNode")

        self.volumeRasToIjk = vtk.vtkMatrix4x4()
        self.volumeIjkToRas = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("InputVolume").GetRASToIJKMatrix(self.volumeRasToIjk)
        self._parameterNode.GetNodeReference("InputVolume").GetIJKToRASMatrix(self.volumeIjkToRas)

    def combineMultipleMasks(self, masks):
        finalMask = np.full(masks[0].shape, False)
        for mask in masks:
            finalMask[mask == True] = True

        return finalMask

    def collectPromptInputsAndPredictSegmentationMask(self):
        if self.currentlySegmenting:
            self.isTherePromptBoxes = False
            self.isTherePromptPoints = False
            currentSliceIndex = self.getIndexOfCurrentSlice()

            # collect prompt points
            positivePromptPointList, negativePromptPointList = [], []

            nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()
            for i in range(nofPositivePromptPoints):
                if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                    if self.sliceAccessorDimension == 2:
                        positivePromptPointList.append([pointIJK[1], pointIJK[2]])
                    elif self.sliceAccessorDimension == 1:
                        positivePromptPointList.append([pointIJK[0], pointIJK[2]])
                    elif self.sliceAccessorDimension == 0:
                        positivePromptPointList.append([pointIJK[0], pointIJK[1]])

            nofNegativePromptPoints = self.negativePromptPointsNode.GetNumberOfControlPoints()
            for i in range(nofNegativePromptPoints):
                if self.negativePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.negativePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [int(round(c)) for c in pointIJK[0:3]]

                    if self.sliceAccessorDimension == 2:
                        negativePromptPointList.append([pointIJK[1], pointIJK[2]])
                    elif self.sliceAccessorDimension == 1:
                        negativePromptPointList.append([pointIJK[0], pointIJK[2]])
                    elif self.sliceAccessorDimension == 0:
                        negativePromptPointList.append([pointIJK[0], pointIJK[1]])

            self.promptPointCoordinations = positivePromptPointList + negativePromptPointList
            self.promptPointLabels = [1] * len(positivePromptPointList) + [0] * len(negativePromptPointList)

            if len(self.promptPointCoordinations) != 0:
                self.isTherePromptPoints = True

            # collect prompt boxes
            boxList = []
            planeList = []

            roiBoxes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
            planes = slicer.util.getNodesByClass("vtkMRMLMarkupsPlaneNode")

            for planeMarkup in planes:
                planeBounds = [0, 0, 0, 0, 0, 0]
                planeMarkup.GetBounds(planeBounds)

                minBoundaries = self.volumeRasToIjk.MultiplyPoint([planeBounds[0], planeBounds[2], planeBounds[4], 1])
                maxBoundaries = self.volumeRasToIjk.MultiplyPoint([planeBounds[1], planeBounds[3], planeBounds[5], 1])

                if currentSliceIndex in self.allPlaneIndices:
                    if self.sliceAccessorDimension == 2:
                        boxList.append([maxBoundaries[1], maxBoundaries[2], minBoundaries[1], minBoundaries[2]])
                    elif self.sliceAccessorDimension == 1:
                        boxList.append([maxBoundaries[0], maxBoundaries[2], minBoundaries[0], minBoundaries[2]])
                    elif self.sliceAccessorDimension == 0:
                        boxList.append([maxBoundaries[0], maxBoundaries[1], minBoundaries[0], minBoundaries[1]])
                

            for roiBox in roiBoxes:
                boxBounds = np.zeros(6)
                roiBox.GetBounds(boxBounds)
                minBoundaries = self.volumeRasToIjk.MultiplyPoint([boxBounds[0], boxBounds[2], boxBounds[4], 1])
                maxBoundaries = self.volumeRasToIjk.MultiplyPoint([boxBounds[1], boxBounds[3], boxBounds[5], 1])

                if self.sliceAccessorDimension == 2:
                    boxList.append([maxBoundaries[1], maxBoundaries[2], minBoundaries[1], minBoundaries[2]])
                elif self.sliceAccessorDimension == 1:
                    boxList.append([maxBoundaries[0], maxBoundaries[2], minBoundaries[0], minBoundaries[2]])
                elif self.sliceAccessorDimension == 0:
                    boxList.append([maxBoundaries[0], maxBoundaries[1], minBoundaries[0], minBoundaries[1]])

            if len(boxList) != 0:
                self.isTherePromptBoxes = True

            # predict mask
            with open(self.featuresFolder + "/" + f"slice_{currentSliceIndex}_features.pkl", "rb") as f:
                if "SAM-2" in self.modelName:
                    self.sam._features = pickle.load(f)
                else:
                    self.sam.features = pickle.load(f)

            if self.isTherePromptBoxes and not self.isTherePromptPoints:
                if "SAM-2" in self.modelName:
                    self.masks, _, _ = self.sam.predict(
                        point_coords=None,
                        point_labels=None,
                        box=np.array(boxList),
                        multimask_output = True,
                    )
                else:
                    inputBoxes = torch.tensor(boxList, device=self.device)
                    transformedBoxes = self.sam.transform.apply_boxes_torch(inputBoxes, self.imageShape)

                    self.masks, _, _ = self.sam.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformedBoxes,
                        multimask_output = True,
                    )
                    self.masks = self.masks.cpu().numpy()
                self.masks = self.combineMultipleMasks(self.masks)

            elif self.isTherePromptPoints and not self.isTherePromptBoxes:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(self.promptPointCoordinations),
                    point_labels=np.array(self.promptPointLabels),
                    multimask_output=True,
                )
                
            elif self.isTherePromptBoxes and self.isTherePromptPoints:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(self.promptPointCoordinations),
                    point_labels=np.array(self.promptPointLabels),
                    box=np.array(boxList[0]),
                    multimask_output=True,
                )

            else:
                self.masks = None

            if self.masks is not None:
                if self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-1":
                    self.producedMask = self.masks[1][:]
                elif self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-2":
                    self.producedMask = self.masks[0][:]
                else:
                    self.producedMask = self.masks[2][:]
            else:
                self.producedMask = np.full(self.imageShape, False)

            qt.QTimer.singleShot(100, self.collectPromptInputsAndPredictSegmentationMask)

    def extractFeatures(self):
        with slicer.util.MessageDialog("Please wait until SAM has processed the input."):
            with slicer.util.WaitCursor():
                self.createSlices()
                self.createFeatures()

        print("Features are extracted. You can start segmentation by placing prompt points or ROIs (boundary boxes)!")


#
# SegmentWithSAMLogic
#


class SegmentWithSAMLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")
