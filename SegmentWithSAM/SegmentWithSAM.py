import logging
import os
import glob
import slicer, qt, vtk, pickle
import numpy as np
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2

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
        self.parent.contributors = ["Zafer Yildiz (Duke University - Mazurowski Lab)"]
        self.parent.helpText = """
SegmentWithSAM aims to asist its users in segmenting medical data on <a href="https://github.com/Slicer/Slicer">3D Slicer</a> by comprehensively integrating the <a href="https://github.com/facebookresearch/segment-anything">Segment Anything Model (SAM)</a> developed by Meta.
See more information in <a href="https://github.com/mazurowski-lab/SlicerSegmentWithSAM">module documentation</a>.
"""
        self.parent.acknowledgementText = """
            This file was originally developed by Zafer Yildiz (Duke University - Mazurowski Lab). 
        """
        slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SegmentWithSAM1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='SegmentWithSAM',
        sampleName='SegmentWithSAM1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'SegmentWithSAM1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='SegmentWithSAM1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='SegmentWithSAM1'
    )

    # SegmentWithSAM2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='SegmentWithSAM',
        sampleName='SegmentWithSAM2',
        thumbnailFileName=os.path.join(iconsPath, 'SegmentWithSAM2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='SegmentWithSAM2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='SegmentWithSAM2'
    )


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
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self) 
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False        
        self.slicesFolder = self.resourcePath("UI") + "/../../../slices"
        self.featuresFolder = self.resourcePath("UI") + "/../../../features"

        self.modelVersion = "vit_h"
        self.modelName = "sam_vit_h_4b8939.pth"
        self.modelCheckpoint = self.resourcePath("UI") + "/../../../" + self.modelName
        self.masks = None

        if not os.path.exists(self.modelCheckpoint):
            print("You need to put SAM checkpoint in " + self.modelCheckpoint + " and restart 3D Slicer!")
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        model = sam_model_registry[self.modelVersion](checkpoint=self.modelCheckpoint)
        model.to(device=self.device)
        self.sam = SamPredictor(model)
        
        self.currentlySegmenting = False
        self.featuresAreExtracted = False
        

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SegmentWithSAM.ui'))
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
        self.ui.goToSegmentEditorButton.connect('clicked(bool)', self.onGoToSegmentEditor)
        self.ui.goToMarkupsButton.connect('clicked(bool)', self.onGoToMarkups)
        self.ui.segmentButton.connect('clicked(bool)', self.onStartSegmentation)
        self.ui.stopSegmentButton.connect('clicked(bool)', self.onStopSegmentButton)
        self.ui.segmentationDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.maskDropDown.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        
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

        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
        
        volumeArray = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("InputVolume"))
        self.volumeShape = volumeArray.shape
        
        if not self._parameterNode.GetNodeReferenceID("positivePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "positive")
            newPromptPointNode.GetDisplayNode().SetSelectedColor(0, 1, 0) 
            self._parameterNode.SetNodeReferenceID("positivePromptPointsNode", newPromptPointNode.GetID())

        if not self._parameterNode.GetNodeReferenceID("negativePromptPointsNode"):
            newPromptPointNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "negative")
            newPromptPointNode.GetDisplayNode().SetSelectedColor(1, 0, 0) 
            self._parameterNode.SetNodeReferenceID("negativePromptPointsNode", newPromptPointNode.GetID())

        if not self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):
            newSegmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'SAM Segmentation')
            self._parameterNode.SetNodeReferenceID("SAMSegmentationNode", newSegmentationNode.GetID())
            newSegmentationNode.CreateDefaultDisplayNodes() 
            newSegmentId = newSegmentationNode.GetSegmentation().AddEmptySegment()
            self._parameterNode.SetParameter("SAMCurrentSegment", newSegmentId)
            
            if newSegmentId not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[newSegmentId] = np.zeros(self.volumeShape)
                print(list(self.segmentIdToSegmentationMask.keys()))

            for i in range(3):
                self.ui.maskDropDown.addItem("Mask-" + str(i+1))

            self._parameterNode.SetParameter("SAMCurrentMask", "Mask-1")

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
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

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        
        self.ui.positivePrompts.setCurrentNode(self._parameterNode.GetNodeReference("positivePromptPointsNode"))
        self.ui.negativePrompts.setCurrentNode(self._parameterNode.GetNodeReference("negativePromptPointsNode"))

        if self._parameterNode.GetNodeReferenceID("SAMSegmentationNode"):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")

            self.ui.segmentationDropDown.clear()
            for i in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
                segmentName = segmentationNode.GetSegmentation().GetNthSegment(i).GetName()
                self.ui.segmentationDropDown.addItem(segmentName)
         
            if self._parameterNode.GetParameter("SAMCurrentSegment"):
                self.ui.segmentationDropDown.setCurrentText(segmentationNode.GetSegmentation().GetSegment(self._parameterNode.GetParameter("SAMCurrentSegment")).GetName())

                if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                    self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)
            
            if self._parameterNode.GetParameter("SAMCurrentMask"):
                self.ui.maskDropDown.setCurrentText(self._parameterNode.GetParameter("SAMCurrentMask"))

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("positivePromptPointsNode", self.ui.positivePrompts.currentNode().GetID())
        self._parameterNode.SetNodeReferenceID("negativePromptPointsNode", self.ui.negativePrompts.currentNode().GetID())
        segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode").GetSegmentation()
        self._parameterNode.SetParameter("SAMCurrentSegment", segmentationNode.GetSegmentIdBySegmentName(self.ui.segmentationDropDown.currentText))
        self._parameterNode.SetParameter("SAMCurrentMask", self.ui.maskDropDown.currentText)

        self._parameterNode.EndModify(wasModified)

    def initializeVariables(self):

        self.volume = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("InputVolume"))
        self.sliceAccessorDimension = self.getSliceAccessorDimension(self.volume.shape)
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

    def createSlices(self):
        if not os.path.exists(self.slicesFolder):
            os.makedirs(self.slicesFolder)
        
        oldSliceFiles = glob.glob(self.slicesFolder + "/*")
        for filename in oldSliceFiles:
            os.remove(filename)
        
        for sliceIndex in range(self.nofSlices):
            sliceImage = self.getSliceBasedOnSliceAccessorDimension(sliceIndex)
            np.save(self.slicesFolder +  "/slice_" + str(sliceIndex), sliceImage)
    
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
            image = np.load(self.slicesFolder + "/" + filename)
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8) 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.sam.set_image(image)
            
            with open(self.featuresFolder + "/" + os.path.splitext(filename)[0] + "_features.pkl", 'wb') as f:
                pickle.dump(self.sam.features, f)

    def onStartSegmentation(self):

        self.initializeVariables()

        currentSliceIndex = self.getIndexOfCurrentSlice()
        previouslyProducedMask = None

        if self.sliceAccessorDimension == 2:
            previouslyProducedMask = self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,:,currentSliceIndex]
        elif self.sliceAccessorDimension == 1:
            previouslyProducedMask = self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,currentSliceIndex,:]
        else:
            previouslyProducedMask = self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][currentSliceIndex,:,:] 
        
        if np.any(previouslyProducedMask):
            segmentationNode = self._parameterNode.GetNodeReference("SAMSegmentationNode")
            currentLabel = segmentationNode.GetSegmentation().GetSegment(self._parameterNode.GetParameter("SAMCurrentSegment")).GetName()

            confirmed = slicer.util.confirmOkCancelDisplay('Are you sure you want to re-annotate ' + currentLabel + ' for the current slice? All of your previous annotation for ' + currentLabel + ' in the current slice will be removed!', windowTitle='Warning')
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
        self.collectPromptInputsAndPredictSegmentationMask()
        self.updateSegmentationScene()
    
    def getSliceAccessorDimension(self, volumeShape):
        if volumeShape[0] != volumeShape[1] and volumeShape[0] != volumeShape[2]:
            return 0
        elif volumeShape[1] != volumeShape[0] and volumeShape[1] != volumeShape[2]:
            return 1

        return 2

    def onStopSegmentButton(self):
        self.currentlySegmenting = False

        for i in range(self.positivePromptPointsNode.GetNumberOfControlPoints()):
            self.positivePromptPointsNode.SetNthControlPointVisibility(i, False)

        for i in range(self.negativePromptPointsNode.GetNumberOfControlPoints()):
            self.negativePromptPointsNode.SetNthControlPointVisibility(i, False)

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
            currentSliceIndex = self.getIndexOfCurrentSlice()

            if self._parameterNode.GetParameter("SAMCurrentSegment") not in self.segmentIdToSegmentationMask:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")] = np.zeros(self.volumeShape)

            if self.sliceAccessorDimension == 2:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,:,currentSliceIndex] = self.producedMask
            elif self.sliceAccessorDimension == 1:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][:,currentSliceIndex,:] = self.producedMask
            else:
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")][currentSliceIndex,:,:] = self.producedMask
                
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                self.segmentIdToSegmentationMask[self._parameterNode.GetParameter("SAMCurrentSegment")],
                self._parameterNode.GetNodeReference("SAMSegmentationNode"),
                self._parameterNode.GetParameter("SAMCurrentSegment"),
                self._parameterNode.GetNodeReference("InputVolume") 
            )
        
        qt.QTimer.singleShot(100, self.updateSegmentationScene)

    def initializeSegmentationProcess(self):

        self.positivePromptPointsNode = self._parameterNode.GetNodeReference("positivePromptPointsNode")
        self.negativePromptPointsNode = self._parameterNode.GetNodeReference("negativePromptPointsNode")
        
        self.volumeRasToIjk  = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("InputVolume").GetRASToIJKMatrix(self.volumeRasToIjk)

    def combineMultipleMasks(self, masks):
        finalMask = np.full(masks[0].shape, False)

        for mask in masks:
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    for k in range(mask.shape[2]):
                        if mask[i][j][k] == True:
                            finalMask[i][j][k] = True

        return finalMask


    def collectPromptInputsAndPredictSegmentationMask(self):
        
        if self.currentlySegmenting:
            self.isTherePromptBoxes = False
            self.isTherePromptPoints = False
            currentSliceIndex = self.getIndexOfCurrentSlice()

            #collect prompt points
            positivePromptPointList, negativePromptPointList = [], []

            nofPositivePromptPoints = self.positivePromptPointsNode.GetNumberOfControlPoints()
            for i in range(nofPositivePromptPoints):
                if self.positivePromptPointsNode.GetNthControlPointVisibility(i):
                    pointRAS = [0, 0, 0]
                    self.positivePromptPointsNode.GetNthControlPointPositionWorld(i, pointRAS)
                    pointIJK = [0, 0, 0, 1]
                    self.volumeRasToIjk.MultiplyPoint(np.append(pointRAS, 1.0), pointIJK)
                    pointIJK = [ int(round(c)) for c in pointIJK[0:3] ]

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
                    pointIJK = [ int(round(c)) for c in pointIJK[0:3] ]

                    if self.sliceAccessorDimension == 2: 
                        negativePromptPointList.append([pointIJK[1], pointIJK[2]])
                    elif self.sliceAccessorDimension == 1:
                        negativePromptPointList.append([pointIJK[0], pointIJK[2]])
                    elif self.sliceAccessorDimension == 0:
                        negativePromptPointList.append([pointIJK[0], pointIJK[1]])
            
            promptPointCoordinations = positivePromptPointList + negativePromptPointList
            promptPointLabels = [1] * len(positivePromptPointList) + [0] * len(negativePromptPointList)
            
            if len(promptPointCoordinations) != 0:
                self.isTherePromptPoints = True

            #collect prompt boxes
            boxList = []
            roiBoxes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")

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
            
            #predict mask
            with open(self.featuresFolder + "/slice_" + str(currentSliceIndex) + "_features.pkl" , 'rb') as f:
                self.sam.features = pickle.load(f)
            
            if self.isTherePromptBoxes and not self.isTherePromptPoints:
                inputBoxes = torch.tensor(boxList, device=self.device)
                transformedBoxes = self.sam.transform.apply_boxes_torch(inputBoxes, self.imageShape)
                
                self.masks, _, _ = self.sam.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformedBoxes,
                    multimask_output=True,
                )

                self.masks = self.masks.cpu().numpy()
                self.masks = self.combineMultipleMasks(self.masks)
                
            elif self.isTherePromptPoints and not self.isTherePromptBoxes:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(promptPointCoordinations), 
                    point_labels=np.array(promptPointLabels), 
                    multimask_output=True
                )

            elif self.isTherePromptBoxes and self.isTherePromptPoints:
                self.masks, _, _ = self.sam.predict(
                    point_coords=np.array(promptPointCoordinations),
                    point_labels=np.array(promptPointLabels),
                    box=np.array(boxList[0]),
                    multimask_output=True,
                )

            else:
                self.masks = None

            if self.masks is not None:
                if self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-1":
                    self.producedMask = self.masks[0][:]
                elif self._parameterNode.GetParameter("SAMCurrentMask") == "Mask-2":
                    self.producedMask = self.masks[1][:]
                else:
                    self.producedMask = self.masks[2][:]
            else:
                self.producedMask = np.full(self.sam.original_size, False)

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

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# SegmentWithSAMTest
#

class SegmentWithSAMTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_SegmentWithSAM1()

    def test_SegmentWithSAM1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('SegmentWithSAM1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SegmentWithSAMLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')