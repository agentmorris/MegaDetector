#!/bin/bash
echo "Starting MegaDetector Docker image"
echo "Current directory:"
echo `pwd`
echo "Python path:"
echo ${PYTHONPATH}
/usr/bin/supervisord
