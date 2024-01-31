cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SegmentWithSAM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/mazurowski-lab/SlicerSegmentWithSAM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Zafer Yildiz (Mazurowski Lab, Duke University)")
set(EXTENSION_DESCRIPTION "SegmentWithSAM aims to asist its users in segmenting medical data on 3D Slicer by comprehensively integrating the Segment Anything Model (SAM) developed by Meta.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main/SegmentWithSAM/Resources/Icons/SegmentWithSAM.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/main/Screenshots/sws4.png")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentWithSAM)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
