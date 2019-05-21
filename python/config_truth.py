#! /usr/bin/env python                                                                                                                                                                                              
from xAODAnaHelpers import Config
c = Config()

c.algorithm("BasicEventSelection", {"m_name": "BasicEventSelector",
                                    "m_isMC" : True,
                                    "m_truthLevelOnly": True,
                                    "m_useMetaData": False
                                    })


c.algorithm("LLPTruthAlgo",{"m_name" : "LLPTruthAlgo",
                            "m_bsm_ContainerName"  : "TruthBSMWithDecayParticles",
                            "m_boson_ContainerName"  : "TruthBoson",
                            "m_jet_ContainerName"  : "AntiKt4TruthDressedWZJets",
                            "m_met_ContainerName"  : "MET_Truth"})

