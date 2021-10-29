#!/bin/bash
DATA="$PWD/Data"
FIGURES="$PWD/Figures"
RESULTS="$PWD/Results"
MODELS="$PWD/Models"
CODE="$PWD"
COMMON="$CODE/common"

export PYTHONPATH="${PYTHONPATH}:$COMMON/TrainingAndTesting:$COMMON/Utils"

export DATA="$DATA"
export FIGURES="$FIGURES"
export MODELS="$MODELS"
export RESULTS="$RESULTS"
export EFFICIENCIES="$RESULTS/Efficiencies"

[ ! -d "$DATA" ] && mkdir -p $DATA
[ ! -d "$FIGURES" ] && mkdir -p $FIGURES
[ ! -d "$MODELS" ] && mkdir -p $MODELS
[ ! -d "$EFFICIENCIES" ] && mkdir -p $EFFICIENCIES
[ ! -d "$RESULTS" ] && mkdir -p $RESULTS