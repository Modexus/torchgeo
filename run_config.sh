#!/bin/bash
CONFIG_FILE="~/torchgeo/conf/tile2vec_naipcdl_train.yaml"
SEED=666
DATA_DIR="/scratch/users/mike"
OUTPUT_DIR="$DATA_DIR/experiments"
LOG_DIR="$DATA_DIR/logs"
OVERWRITE=false
ARGS=""

while [[ $# -gt 0 ]]
do
        case $1 in
                -c|--config_file)
                CONFIG_FILE=$2
                shift
                shift
                ;;
                -S|--seed)
                SEED=$2
                shift
                shift
                ;;
                -o|--output_dir)
                OUTPUT_DIR=$2
                shift
                shift
                ;;
                -l|--log_dir)
                LOG_DIR=$2
                shift
                shift
                ;;
                -O|--overwrite)
                OVERWRITE=$2
                shift
                shift
                ;;
                *)
                ARGS+=$1" " 
                shift
                ;;
        esac
done

if [ -f "$CONFIG_FILE" ]; then
        echo -e "Found configuration file $CONFIG_FILE"
else
        echo -e "Error: No configuration file found at $CONFIG_FILE"
        exit 1
fi

if [ -d "$OUTPUT_DIR" ]; then
        echo -e "Found output dir $OUTPUT_DIR"
else
        echo -e "Error: No output dir found at $OUTPUT_DIR"
        exit 1
fi

if [ -d "$DATA_DIR" ]; then
        echo -e "Found data dir $DATA_DIR"
else
        echo -e "Error: No data dir found at $DATA_DIR"
        exit 1
fi

if [ -d "$LOG_DIR" ]; then
        echo -e "Found log dir $LOG_DIR"
else
        echo -e "Error: No log dir found at $LOG_DIR"
        exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    SEED=666
fi

if [ "$OVERWRITE" != false ]; then
    if [ "$OVERWRITE" != true ]; then
        echo -e "Error: Overwrite value not boolean $OVERWRITE"
        exit 1
    fi
fi
echo -e "Overwrite value set to $OVERWRITE"

echo "Args:"
ARGS_ARR=( ${ARGS} )
printf "'%s'\n" "${ARGS_ARR[@]}"

cd ~/torchgeo
python ~/torchgeo/train.py config_file="${CONFIG_FILE}" program.seed="${SEED}" program.output_dir="${OUTPUT_DIR}" program.data_dir="${DATA_DIR}" program.log_dir="${LOG_DIR}" program.overwrite="${OVERWRITE}" ${ARGS}