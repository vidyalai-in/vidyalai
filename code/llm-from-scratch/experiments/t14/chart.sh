#!/bin/bash
# usage: chart.sh <name>: run diagrams/<name>.py, then render the light SVG to PNG with sharp at density 144
cd /home/amit/AlpsCode/ai/vidyalai
python3 diagrams/$1.py public/diagrams/$1 && node -e "require('sharp')('public/diagrams/$1.svg',{density:144}).png().toFile('public/diagrams/$1.png').then(()=>console.log('ok $1'))"
node -e "require('sharp')('public/diagrams/$1-dark.svg',{density:144}).png().toFile('/home/amit/.cache/vidyalai-scratch/t14/post/$1-dark.png')"
