#!/usr/bin/env bash

# Format the bib file according to DOI rules. Taken from Janek Bevendorff.
# Requires the `bibtool` and `moreutils` packages.

set -e

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

FILENAME=
OUTPUTFILE=
CHECK=0

show_help() {
    echo "Format a bib file."
    echo ""
    echo "Usage:"
    echo "${0} --file INPUTFILE [--output OUTPUTNAME] [--check]"
    echo ""
    echo "If no output file name is specified, the input file will be overwritten."
    echo ""
    echo "If check is set, the programm will return with an error if the file changed."
    exit 0
}

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        -f|--file)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                FILENAME=$2
                shift
            else
                die 'ERROR: "--file" requires a non-empty option argument.'
            fi
            ;;
        --file=?*)
            FILENAME=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --file=)         # Handle the case of an empty --file=
            die 'ERROR: "--file" requires a non-empty option argument.'
            ;;
        -o|--output)
            if [ "$2" ]; then
                OUTPUTFILE=$2
                shift
            else
                die 'ERROR: "--output" requires a non-empty option argument.'
            fi
            ;;
        --output=?*)
            OUTPUTFILE=${1#*=}
            ;;
        --output=)
            die 'ERROR: "--output" requires a non-empty option argument.'
            ;;
        -c|--check)
            CHECK=1
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

if [[ -z ${FILENAME} ]]; then
    die 'No input file given.'
fi

if [[ -z ${OUTPUTFILE} ]]; then
    OUTPUTFILE="${FILENAME}"
fi

if [ "${CHECK}" = 1 ]; then
    HASH=$(md5sum ${FILENAME} | cut -d ' ' -f 1)
fi

bibtool -- 'print.align = 24' -- 'print.align.key = 0' \
        -- 'print.equal.right = off' -- 'pass.comments = on' \
        -- 'print.line.length = 1000' -- 'print.use.tab = off' \
        -- 'rewrite.rule = {doi# "https?://.*doi.*\.org/\(10\.[0-9]+/.+\)"# "\1"}' \
        -d \
        -s \
        ${FILENAME} | sed '/./,$!d' | sponge ${OUTPUTFILE}


if [ "${CHECK}" = 1 ]; then
    HASH_OUT=$(md5sum ${OUTPUTFILE} | cut -d ' ' -f 1)
    if [ ${HASH} != ${HASH_OUT} ]; then
        die 'File changed'
    fi
fi

exit 0