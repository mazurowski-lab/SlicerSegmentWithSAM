#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentWithSAM)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )
set(EXTENSION_HOMEPAGE "https://github.com/mazurowski-lab/SlicerSegmentWithSAM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Zafer Yildiz (Mazurowski Lab, Duke University)")
set(EXTENSION_DESCRIPTION "SegmentWithSAM aims to asist its users in segmenting medical data on 3D Slicer by comprehensively integrating the Segment Anything Model (SAM) developed by Meta.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/SegmentWithSAM/Resources/Icons/SegmentWithSAM.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentWithSAM/Screenshots/sws1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies
#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  # NA
endif()
