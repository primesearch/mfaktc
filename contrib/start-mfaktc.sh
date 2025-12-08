#!/bin/bash

# This file is part of mfaktc.
# Copyright (c) 2019-2025  Danny Chia
# Copyright (c) 2009-2011  Oliver Weihe (o.weihe@t-online.de)
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

# start-mfaktc.sh
#
# Enables users to easily launch multiple mfaktc instances on multi-GPU
# systems.
#
# Usage:
#   ./start-mfaktc.sh [device ID]
#
# Device ID provided:
#   Uses symbolic links to launch an mfaktc instance from a device-specific
#   folder. You will need to manually add a worktodo.txt file each time you
#   start a new batch of assignments.
#
#   Device-specific settings are supported. You can simply add an mfaktc.ini
#   file to the device-specific folder as the script will not attempt to
#   overwrite or delete a file that already exists.
#
# Device ID not provided:
#   Launches the mfaktc executable from the root folder. Default behavior.
#
# To prevent concurrency issues, the script enforces a limit of one mfaktc
# instance per GPU. It is recommended to set SieveOnGPU=1 in mfaktc.ini so
# that each instance can make full use of a device.

APP=mfaktc
APP_SETTINGS=$APP.ini
LOCK=$APP.lock

run_on_device() {
    # ensure instance has its own folder
    ! test -e "device-$1" && mkdir "device-$1"
    if ! cd "device-$1"
    then
        echo "error: could not enter directory 'device-$1' for device $1"
        exit 1
    fi

    # don't run if device is in use
    if [[ -e $LOCK ]]; then
        echo "error: lock file $LOCK exists, mfaktc may already be running on device $1"
        exit 1
    else
        touch $LOCK
    fi

    # create symbolic links
    if [[ -e $APP ]]; then
        echo "error: cannot create symbolic link '$APP' in directory 'device-$1' as a file"
        echo "       with that name already exists. Stopped to prevent potential data loss"
        exit 1
    fi
    ln -s ../$APP .

    # don't overwrite custom settings
    ! test -e $APP_SETTINGS && ln -s ../$APP_SETTINGS .

    # run mfaktc on specified device
    ./$APP -d "$1"

    # clean up
    rm -f $APP $LOCK

    # don't delete mfaktc.ini unless it's a symbolic link
    test -L $APP_SETTINGS && rm $APP_SETTINGS
}

if [ "$#" == 0 ]
then
    # don't run if device is in use
    if [[ -e $LOCK ]]; then
        echo "error: lock file $LOCK exists, mfaktc may already be running"
        exit 1
    else
        touch $LOCK
    fi

    # run mfaktc on default device
    ./$APP

    # clean up
    rm $LOCK
else
    run_on_device "$1"
fi
