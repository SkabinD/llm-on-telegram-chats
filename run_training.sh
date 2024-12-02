#!/bin/bash

config=""

usage() {
    echo "Usage: $0 [-c <config>]"
    echo "  -c <config> Specify a configuration file"
    exit 1
}

while getopts ":c:" opt; do
    case ${opt} in
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

if [ -z "$config" ]; then
    echo "Error: Configuration file not specified." >&2
    usage
fi

python scripts/train/"$config".py