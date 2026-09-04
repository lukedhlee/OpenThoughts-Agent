#!/bin/bash
# jrc.sh '<cmd>' — run a command on the JURECA login node through the Jupiter-login ControlMaster.
ssh -o ConnectTimeout=25 jupiter "ssh -o BatchMode=yes -S ~/.ssh/cm_jureca/qwen36 jureca05.fz-juelich.de $(printf '%q' "$1")"
