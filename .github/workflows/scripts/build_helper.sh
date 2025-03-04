#!/bin/bash
#
# This file is part of mfaktc.
# Copyright (c) 2025        NStorm (https://github.com/N-Storm)
# Copyright (c) 2009-2011   Oliver Weihe (o.weihe@t-online.de)
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
#
# build_helper.sh
# This script is used for continuous integration in GitHub Actions. It gathers
# information on software versions and saves it to the file build_helper.sh.out
# for later use by GitHub Actions workflows. It also patches makefiles to
# support building under GitHub Actions runner environments and compiling
# kernels for all devices with NVCC-supported compute capabilities.

if [[ -z "$1" ]]; then
  echo "Usage: $0 <CUDA version>" >&2
  exit 1
fi

# Windows may default to its own 'sort' in the PATH variable, use the full path
# to GNU sort
export GSORT='/usr/bin/sort'

declare -a CUDA_VERSION
CUDA_VERSION=( $(echo "$1" | head -n1 | grep -Eom1 -e '^[1-9]([0-9])?\.[0-9]{1,2}(\.[0-9]{1,3})?$' | tr '.' ' ') )
if [[ -z "${CUDA_VERSION[*]}" ]]; then
  echo "ERROR! Can't parse CUDA version $1" >&2
  exit 2
fi

CUDA_VER_MAJOR=${CUDA_VERSION[0]}
CUDA_VER_MINOR=${CUDA_VERSION[1]}
CUDA_VER="${CUDA_VER_MAJOR}${CUDA_VER_MINOR}"
echo -e "CUDA_VER_MAJOR=${CUDA_VER_MAJOR}\nCUDA_VER_MINOR=${CUDA_VER_MINOR}" > "$0.out"

# CUDA supports the --list-gpu-arch flag from 11.0.0 onwards.
# For older CUDA versions, use grep to parse the supported architectures from
# the output of --help
[ $CUDA_VER -gt 110 ] && NVCC_OPTS='--list-gpu-arch' || NVCC_OPTS='--help'
NVCC_REGEX='compute_[1-9][0-9]{1,2}'
# CUDA 11.0.x is a special case. Its --help output lists compute_32 and higher,
# but only compute capability 3.5 and later are supported.
[ $CUDA_VER -eq 110 ] && NVCC_REGEX='compute_(3[5-9]|[4-9][0-9])'

declare -a CC_LIST
CC_LIST=( $(nvcc $NVCC_OPTS | grep -Eoe "$NVCC_REGEX" | cut -d '_' -f2 | $GSORT -un | xargs) )
if [ ${#CC_LIST[*]} -eq 0 ]; then
  echo "Error: could not parse list of supported compute capabilities" >&2
  exit 3
elif [ ${#CC_LIST[*]} -lt 3 ]; then
  echo "Warning: less than three (3) supported compute capabilities" >&2
fi

echo "All supported CCs: ${CC_LIST[*]}, CC_MIN=${CC_LIST[0]}, CC_MAX=${CC_LIST[-1]}"
echo -e "CC_LIST=\"${CC_LIST[*]}\"\nCC_MIN=${CC_LIST[0]}\nCC_MAX=${CC_LIST[-1]}" >> "$0.out"

echo 'Removing NVCCFLAGS strings with "arch=..." entries from makefiles and populating them with discovered supported values.'
sed -i '/^NVCCFLAGS += --generate-code arch=compute.*/d' src/Makefile.win src/Makefile
for CC in "${CC_LIST[@]}"; do
  sed -i "/^NVCCFLAGS = .*\$/a NVCCFLAGS += --generate-code arch=compute_${CC},code=sm_${CC}" src/Makefile src/Makefile.win
done

if [ $CUDA_VER -ge 110 ]; then
  echo 'Adding NVCCFLAGS to allow unsupported MSVC versions...'
  sed -i '/^NVCCFLAGS = .*/a NVCCFLAGS += -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' src/Makefile.win
fi
if [ $CUDA_VER -lt 120 ]; then
  echo "Adding libraries to LDFLAGS to support static build on older Ubuntu versions..."
  sed -i -E 's/^(LDFLAGS = .*? -lcudart_static) (.*)/\1 -ldl -lrt -lpthread \2/' src/Makefile
fi

echo 'Gathering version info on generic compiler and NVCC...'
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
fi

if [[ -x "$(command -v powershell.exe)" ]]; then
  OS_VER="$(powershell -Command "[System.Environment]::OSVersion.VersionString")"
fi

NVCC_VER="$(nvcc --version | tail -n1 | sed -E 's/^Build //')"

echo -e "COMPILER_VER=$COMPILER_VER\nNVCC_VER=$NVCC_VER\nOS_VER=${OS_VER}" | tee -a $0.out
