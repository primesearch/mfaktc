#!/bin/bash
#
# build_helper.sh
# This script are used by the CI/CD builds with Github Actions workflow.
# It gathers some info on software versions and saves it to file 
# build_helper.sh.out which are later used by the actions workflow.
# It also patches Makefiles to support building under Github Action
# runners environment and compile kernels for every compute capability
# device supported by the NVCC.
#
# Copyright (c) 2025 NStorm (https://github.com/N-Storm/)
# This file is part of mfaktc.
# Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
#
# mfaktc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mfaktc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

if [[ -z "$1" ]]; then
  echo "Usage: $0 <CUDA version>" >&2
  exit 1
fi

# Windows may have it's sort first on PATH, this is the shortcut
# to call GNU sort by the full path
export GSORT='/usr/bin/sort'

CUDA_VERSION_FULL="$(echo "$1" | head -n1 | grep -Eom1 -e '^[1-9]([0-9])?\.[0-9]{1,2}(\.[0-9]{1,3})?$')"
declare -a CUDA_VERSION
CUDA_VERSION=( $(echo "$CUDA_VERSION_FULL" | tr '.' ' ') )
if [[ -z "${CUDA_VERSION[*]}" ]]; then
  echo "ERROR! Can't parse CUDA version $1" >&2
  exit 2
fi

CUDA_VER_MAJOR=${CUDA_VERSION[0]}
CUDA_VER_MINOR=${CUDA_VERSION[1]}
CUDA_VER="${CUDA_VER_MAJOR}${CUDA_VER_MINOR}"
echo -e "CUDA_VER_MAJOR=${CUDA_VER_MAJOR}\nCUDA_VER_MINOR=${CUDA_VER_MINOR}" > "$0.out"

# Starting from 11.0.0 CUDA has --list-gpu-arch flag.
# For older versions we have to grep out supported CC versions from help.
[ $CUDA_VER -gt 110 ] && NVCC_OPTS='--list-gpu-arch' || NVCC_OPTS='--help'
NVCC_REGEX='compute_[1-9][0-9]{1,2}'
# Special case with CUDA 11.0.x. It's help lists compute_32 and higher, but only CCs from 35 are supported.
[ $CUDA_VER -eq 110 ] && NVCC_REGEX='compute_(3[5-9]|[4-9][0-9])'

declare -a CC_LIST
CC_LIST=( $(nvcc $NVCC_OPTS | grep -Eoe "$NVCC_REGEX" | cut -d '_' -f2 | $GSORT -un | xargs) )
if [ ${#CC_LIST[*]} -eq 0 ]; then
  echo "ERROR! Unable to parse a list of CCs" >&2
  exit 3
elif [ ${#CC_LIST[*]} -lt 3 ]; then
  echo "WARN Number of supported CC versions less than 3" >&2
fi

echo "All supported CCs: ${CC_LIST[*]}, CC_MIN=${CC_LIST[0]}, CC_MAX=${CC_LIST[-1]}"
echo -e "CC_LIST=\"${CC_LIST[*]}\"\nCC_MIN=${CC_LIST[0]}\nCC_MAX=${CC_LIST[-1]}" >> "$0.out"

echo 'Removing NVCCFLAGS strings with CC arch entries from the Makefile & Makefile.win and populating with discovered supported values.'
sed -i '/^NVCCFLAGS += --generate-code arch=compute.*/d' src/Makefile.win src/Makefile
for CC in "${CC_LIST[@]}"; do
  sed -i "/^NVCCFLAGS = .*\$/a NVCCFLAGS += --generate-code arch=compute_${CC},code=sm_${CC}" src/Makefile src/Makefile.win
done

if [ $CUDA_VER -ge 110 ]; then
  echo 'Adding NVCCFLAGS to allow unsupported MSVC compiler versions...'
  sed -i '/^NVCCFLAGS = .*/a NVCCFLAGS += -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' src/Makefile.win
fi
if [ $CUDA_VER -lt 120 ]; then
  echo "Adding libraries to LDFLAGS to support static build on older Ubuntu versions..."
  sed -i -E 's/^(LDFLAGS = .*? -lcudart_static) (.*)/\1 -ldl -lrt -lpthread \2/' src/Makefile
fi

echo 'Gathering version info on generic compiler and nvcc...'
if [[ -x "$(command -v vswhere.exe)" ]]; then
  CC_VSPROD="$(vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName)"
  COMPILER_VER="${CC_VSPROD}, $(vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion)"
elif [[ -x "$(command -v powershell.exe)" ]]; then
  CC_VSINFO="$(powershell -Command Get-VSSetupInstance)"
  CC_VSPROD="$(echo "$CC_VSINFO" | grep DisplayName | cut -d':' -f2 | xargs)"
  COMPILER_VER="${CC_VSPROD}, $(echo "$CC_VSINFO" | grep InstallationVersion | cut -d':' -f2 | xargs)"
else
  COMPILER_VER="$(gcc --version | head -n1)"
  source /etc/os-release
  OS_VER="${PRETTY_NAME}"
  OS_TYPE="linux64"
fi

if [[ -x "$(command -v powershell.exe)" ]]; then
  OS_VER="$(powershell -Command "[System.Environment]::OSVersion.VersionString")"
  OS_TYPE="win64"
fi

NVCC_VER="$(nvcc --version | tail -n1 | sed -E 's/^Build //')"

# Version from src/params.h
MFAKTC_VER="$(grep -Eo '#define MFAKTC_VERSION "([0-9]\.[0-9]+\.[-0-9a-z]+)"' src/params.h | cut -d '"' -f 2)"

# Git-formatted version
GIT_TAG_VER="$(git describe --tags)"

# Compare MFAKTC_VER with GIT_TAG_VER up to a length of MFAKTC_VER.
# If they don't match, warn and use MFAKTC_VER and short commit hash for BASE_NAME.
# Otherwise, GIT_TAG_VER will be used, which should include version, short hash and a number of commits
# since last tag when git HEAD isn't directly referenced by a tag. Or just a tag when current commit has tag
# reference. This gives shorter BASE_NAME without commit hash for releases.
if [[ "$MFAKTC_VER" != "${GIT_TAG_VER:0:${#MFAKTC_VER}}" ]]; then
  SHA_SHORT="$(git rev-parse --short HEAD)"
  BASE_NAME="mfaktc-${MFAKTC_VER}-${SHA_SHORT}-${OS_TYPE}-cuda${CUDA_VERSION_FULL}"
  echo "Warning: version from git describe (${GIT_TAG_VER}) doesn't begins with MFAKTC_VER (${MFAKTC_VER}) from params.h"
  echo "Using version from params.h and short commit hash (${SHA_SHORT}) for BASE_NAME"
else
  BASE_NAME="mfaktc-${GIT_TAG_VER}-${OS_TYPE}-cuda${CUDA_VERSION_FULL}"
fi

echo -e "COMPILER_VER=${COMPILER_VER}\nNVCC_VER=${NVCC_VER}\nOS_VER=${OS_VER}\nBASE_NAME=${BASE_NAME}" | tee -a "$0.out"
