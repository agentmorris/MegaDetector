if [[ $(uname -m) == 'arm64' ]]; then
    echo "Installing MegaDetector environment for Apple silicon"
mamba env create --file envs/environment-detector-m1.yml
else
    echo "Installing MegaDetector environment for Intel Macs"
    mamba env create --file envs/environment-detector-mac.yml
fi

