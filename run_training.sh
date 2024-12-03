#!/bin/bash

script=""
config=""

usage() {
    echo "Usage: $0 [-s <script>] [-c <config>]"
    echo "  -s <config> Specify a train script file"
    echo "  -c <config> Specify a train configuration file"
    exit 1
}

while getopts ":s:c:" opt; do
    case ${opt} in
        s ) 
            script=$OPTARG
            ;;
        c )
            config=$OPTARG
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        : )
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

shift $((OPTIND -1))

if [ -z "$config" ] || [ -z "$script" ]; then
    echo "Error: Train script or train configuration file is not specified" >&2
    usage
fi

python scripts/train/"$script.py" --config "$config.yaml"