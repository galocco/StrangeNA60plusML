#!/bin/bash

[ -z "$ALICE_PHYSICS" ] && echo "AliPhysics environment required! Load it before sourcing this." && return

HYPERML_DATA="$PWD/Data"
HYPERML_FIGURES="$PWD/Figures"
HYPERML_RESULTS="$PWD/Results"
HYPERML_MODELS="$PWD/Models"
HYPERML_CODE="$PWD"
HYPERML_COMMON="$HYPERML_CODE/common"

export PYTHONPATH="${PYTHONPATH}:$HYPERML_COMMON/TrainingAndTesting:$HYPERML_COMMON/Utils"
BODY_2=0
BODY_3=0

if [ $# -eq 0 ]; then
      BODY_2=1
      BODY_3=1
fi

if [ "$1" == "2" ] || [ "$1" == "2body" ] || [ "$1" == "2Body" ]; then
      BODY_2=1
fi
if [ "$1" == "3" ] || [ "$1" == "3body" ] || [ "$1" == "3Body" ]; then
      BODY_3=1
fi

if [ $BODY_2 -eq 1 ]; then    
      export HYPERML_DATA_2="$HYPERML_DATA/2Body"
      export HYPERML_FIGURES_2="$HYPERML_FIGURES/2Body"
      export HYPERML_MODELS_2="$HYPERML_MODELS/2Body"
      export HYPERML_RESULTS_2="$HYPERML_RESULTS/2Body"
      export HYPERML_EFFICIENCIES_2="$HYPERML_RESULTS_2/Efficiencies"

      [ ! -d "$HYPERML_DATA_2" ] && mkdir -p $HYPERML_DATA_2
      [ ! -d "$HYPERML_FIGURES_2" ] && mkdir -p $HYPERML_FIGURES_2
      [ ! -d "$HYPERML_MODELS_2" ] && mkdir -p $HYPERML_MODELS_2
      [ ! -d "$HYPERML_EFFICIENCIES_2" ] && mkdir -p $HYPERML_EFFICIENCIES_2
      [ ! -d "$HYPERML_RESULTS_2" ] && mkdir -p $HYPERML_RESULTS_2
fi

if [ $BODY_3 -eq 1 ]; then    
      export HYPERML_DATA_3="$HYPERML_DATA/3Body"
      export HYPERML_FIGURES_3="$HYPERML_FIGURES/3Body"
      export HYPERML_MODELS_3="$HYPERML_MODELS/3Body"
      export HYPERML_EFFICIENCIES_3="$HYPERML_EFFICIENCIES/3Body"
      export HYPERML_RESULTS_3="$HYPERML_RESULTS/3Body"
      export HYPERML_EFFICIENCIES_3="$HYPERML_RESULTS_3/Efficiencies"
fi
