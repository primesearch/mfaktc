#!/bin/bash

# This file is part of mfaktc.
# Copyright (C) 2009-2025  Oliver Weihe (o.weihe@t-online.de)
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

case "${1}" in
  win)
    OBJ="obj"
    ;;
  *)
    OBJ="o"
    ;;
esac

(
# need to define CUDA runtime version during cpp runs because we ignore
# system includes.

# special code for mfaktc.c that includes either selftest-data-mersenne.c or
# selftest-data-wagstaff.c, depending on whether mfaktc is configured for
# Mersenne or Wagstaff numbers in params.h. For simplicity's sake, we just add
# both files.
cpp -D CUDART_VERSION=6050 -include selftest-data-mersenne.c -include \
    selftest-data-wagstaff.c -MM mfaktc.c
echo

for FILE in checkpoint.c output.c parse.c read_config.c sieve.c \
            signal_handler.c timer.c tf_96bit.cu tf_barrett96.cu \
            tf_barrett96_gs.cu gpusieve.cu cuda_utils.cu crc.c
do
  cpp -D CUDART_VERSION=6050 -MM ${FILE}
  echo
done

# special case for 75-bit kernels
cpp -D CUDART_VERSION=6050 -MM tf_96bit.cu -MT tf_75bit.o
) | sed s@\.o:@\.${OBJ}:@
