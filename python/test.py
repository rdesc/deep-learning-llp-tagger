#! /usr/bin/env python                                                                                                                                                                                              
from xAODAnaHelpers import Config
c = Config()

c.algorithm("BasicEventSelection", {"m_name": "BasicEventSelector",
                                    "m_isMC" : False,
                                    "m_truthLevelOnly": False,
                                    "m_useMetaData": False,
                                    "m_doPUreweighting": False ,
                                    "m_PRWFileNames": "${WorkDir_DIR}/data/FactoryTools/DV/rpvll_DV.prw.root",
                                    "m_applyGRLCut": True,
                                    "m_GRLxml": "${WorkDir_DIR}/data/FactoryTools/DV/data16_13TeV.periodAllYear_DetStatus-v83-pro20-15_DQDefects-00-02-04_PHYS_StandardGRL_All_Good_25ns_DAOD_RPVLL_r8669.xml",
                                    "m_applyPrimaryVertexCut": False,
                                    "m_applyTriggerCut": False,
                                    "m_storeTrigDecisions": True,
                                    "m_triggerSelection": "HLT_j30_muvtx_noiso",
                                    "m_storePassL1": False,
                                    "m_storePassHLT": False,
                                    "m_storeTrigKeys": False })

c.algorithm("TreeAlgo", {"m_name": "tree",
                         "m_evtDetailStr": "pileup eventCleaning truth",
                         "m_trigDetailStr": "basic menuKeys passTriggers",
                         "m_jetContainerName": "AntiKt4EMTopoJets",
                         "m_jetDetailStr": "kinematic substructure rapidity energy JVT clean",
                         "m_METReferenceContainerName": "MET_Reference_AntiKt4EMTopo",
                         "m_METDetailStr": "metTrk",
                         "m_trackParticlesContainerName": "InDetTrackParticles",
                         "m_trackParticlesDetailStr": "kinematic vertex",
                         "m_muContainerName": "Muons",
                         "m_muDetailStr": "kinematic"})

c.algorithm("StoreMSVX", {"m_name": "StoreMSVX"})

c.algorithm("CalculateDiVertVars", {"m_name": "Calculator"})

c.algorithm("WriteOutputNtuple", {"m_name": "NtupleWriter",
                                  "outputName":"tree",
                                  "regionName":"DiVert"})
