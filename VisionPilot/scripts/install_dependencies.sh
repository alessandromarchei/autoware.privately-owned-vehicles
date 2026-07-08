#!/usr/bin/env bash

set -e

###############################################################################
# Enable Ubuntu repositories
###############################################################################

sudo apt update
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe

###############################################################################
# Install optimization libraries
###############################################################################

echo "Installing IPOPT and CppAD..."

sudo apt install -y \
    coinor-libipopt-dev \
    cppad

# Ubuntu installs IPOPT headers under /usr/include/coin,
# while CppAD expects /usr/include/coin-or.
# Create the compatibility symlink only if needed.

if [ ! -e /usr/include/coin-or ]; then
    echo "Creating /usr/include/coin-or symbolic link..."
    sudo ln -s /usr/include/coin /usr/include/coin-or
fi

###############################################################################
# Install WebRTC / GStreamer dependencies
###############################################################################

echo "Installing GStreamer/WebRTC dependencies..."

sudo apt install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libsoup2.4-dev \
    libjson-glib-dev
