#################
# mfaktc README #
#################

Table of contents

0   About mfaktc
1   Supported hardware
2   Compilation
2.1 Linux
2.2 Windows
3   Running mfaktc
3.1 Linux
3.2 Windows
4   Getting work and reporting results
5   Known issues
5.1 Non-issues
6   Tuning
7   FAQ
8   .plan


##################
# 0 About mfaktc #
##################

mfaktc is a program that trial factors Mersenne numbers for the Great Internet
Mersenne Prime Search. It stands for "Mersenne faktorisation* with CUDA" and
was written for Nvidia GPUs. mfaktc runs almost entirely on the GPU in version
0.20 and later.

Primality tests are computationally intensive, but we can save time by finding
small factors. GPUs are very efficient at this task due to their parallel
nature. Only one factor is needed to prove a number composite.

mfaktc uses a modified Sieve of Eratosthenes to generate a list of possible
factors for a given Mersenne number. It then uses modular exponentiation to
test these factors. You can find more details at the GIMPS website:
https://mersenne.org/various/math.php#trial_factoring

* portmanteau of the English word "factorisation" and the German word
"Faktorisierung"


########################
# 1 Supported hardware #
########################

mfaktc should run on all CUDA-capable Nvidia GPUs with Compute Capability 1.1
and above. To my knowledge, the only GPU with Compute Capability 1.0 is the G80
chip in the GeForce 8800 Ultra / GTX / GTS 640 / GTS 320 and their Quadro and
Tesla variants.

For AMD GPUs, there is an OpenCL port of mfaktc by Bertram Franz called mfakto:
https://github.com/primesearch/mfakto

Important note: mfaktc will no longer support compute capability 1.x devices
in version 0.24 onwards.


#################
# 2 Compilation #
#################

General requirements:
- CUDA Toolkit
  - see https://developer.nvidia.com/cuda-toolkit for download and installation
    instructions
- C compiler

Some compile-time settings in the file src/params.h can be changed:
- in the first section are settings which "advanced users" can change if they
  think it is beneficial. These settings have been verified for reasonable
  values.
- in the middle are debug options which can be enabled. These are only useful
  for debugging purposes.
- the last part contains defines which should *not* be changed unless you
  fully understand them. It is possible to easily screw something up.

Be aware that mfaktc 0.24.0 and CUDA Toolkit 12.2 drop support for 32-bit
builds. You will need to use the '0.23' branch and an older CUDA Toolkit to
compile mfaktc for 32 bits. See this thread for details:
https://forums.developer.nvidia.com/t/whats-the-last-version-of-the-cuda-toolkit-to-support-32-bit-applications/323106/4

In any case, a 64-bit build is preferred except on some old low-end GPUs.
Testing on an Intel Core i7 CPU has shown that the performance-critical CPU
code runs about 33% faster compared to 32 bits.

It should be noted that each CUDA version only supports specific compute
capabilities. You may have to enable or disable certain flags in the makefile
before compiling mfaktc, or nvcc may exit due to an "unsupported gpu
architecture" error.

Use this table to determine which compute capabilities are supported:
https://en.wikipedia.org/wiki/CUDA#GPUs_supported

#############
# 2.1 Linux #
#############

Steps:
- navigate to the mfaktc root folder
- cd src
- open the makefile and verify CUDA_DIR points to the CUDA installation
  - for 32-bit builds: also change "lib64" to "lib" in CUDA_LIB
- optional: run "make clean" to remove any build artifacts
- make

If compilation succeeds, the binary "mfaktc" should appear in the root folder.

mfaktc was originally compiled with:
- OpenSUSE 12.2 x86_64
- gcc 4.7.1
- Nvidia driver 343.36
- CUDA Toolkit 6.5

It should be possible to use an older CUDA Toolkit to build mfaktc. However,
this may not give the best performance.


###############
# 2.2 Windows #
###############

OS-specific requirements:
- Microsoft Visual Studio
- make for Windows

mfaktc was originally built using Visual Studio 2012 Professional on a 64-bit
Windows 7 machine. However, these instructions will work for all recent Windows
and Visual Studio versions.

You will need a GNU-compatible version of make as the makefiles are not
compatible with nmake. GnuWin32 provides a native port for 32-bit Windows:
https://gnuwin32.sourceforge.net/packages/make.htm

You can add C:\Program Files (x86)\GnuWin32\bin to the Path system variable so
that make is always available.

Before attempting to compile mfaktc, be sure the system variable CUDA_PATH
points to your CUDA installation. In most cases, the CUDA installer should
automatically set this variable.


Steps:
- open the Visual Studio Developer Command Prompt for the desired architecture
  - for 64 bits: x64 Native Tools Command Prompt
  - for 32 bits: x86 Native Tools Command Prompt
- navigate to the mfaktc root folder
- cd src
- optional: clean up any build artifacts
  - for 64 bits: make -f Makefile.win clean
  - for 32 bits: make -f Makefile.win32 clean
- start the build
  - for 64 bits: make -f Makefile.win
  - for 32 bits: make -f Makefile.win32

You should see the binary "mfaktc-win-64.exe" or "mfaktc-win-32.exe" in the
mfaktc root folder.


####################
# 3 Running mfaktc #
####################

Just run 'mfaktc -h' to see what parameters it accepts. You can also check
mfaktc.ini for additional options and a short description of each one. mfaktc
typically fetches assignments from a worktodo.txt file, but this can be
customized. See section 4 for steps to obtain assignments.

mfaktc has built-in self-test that checks for errors. Please run the full
self-test each time you:
- recompile the code
- download a new binary from somewhere
- update the Nvidia driver
- change the hardware

A typical worktodo.txt file looks like this:
-- begin example --
Factor=[assignment ID],66362159,64,68
Factor=[assignment ID],3321932839,50,71
-- end example --

You can launch mfaktc after getting the assignments. In this case, mfaktc
should trial factor M66362159 from 64 to 68 bits, and then M3321932839 from 50
to 71 bits.


#############
# 3.1 Linux #
#############

- build mfaktc using the above instructions or download a stable release
- go to the mfaktc root folder and run "./mfaktc"

###############
# 3.2 Windows #
###############

mfaktc works very similarly on Windows. You can just run "mfaktc-win-64" or
"mfaktc-win-32" in Command Prompt (cmd.exe) to launch the executable, or simply
double-click it in File Explorer.

However, you do need to prepend the executable name with ".\" in PowerShell or
Windows Terminal.


########################################
# 4 Getting work and reporting results #
########################################

You must have a PrimeNet account to participate. Simply go to the GIMPS website
at https://mersenne.org and click "Register" to create one. Once you've signed
up, you can get assignments in several ways.

Using the AutoPrimeNet application:
    AutoPrimeNet allows clients that do not natively support PrimeNet to obtain
    work and submit results. It is recommended to use this tool when possible.
    See the AutoPrimeNet download page for instructions:
    https://download.mersenne.ca/AutoPrimeNet

From the GIMPS website:
    Step 1) log in to the GIMPS website with your username and password
    Step 2) on the menu bar, select Manual Testing > Assignments
    Step 3) open the link to the manual GPU assignment request form
    Step 4) enter the number of assignments or GHz-days you want
    Step 5) click "Get Assignments"

    Users with older GPUs may want to use the regular form.

Using the GPU to 72 website:
    GPU to 72 "subcontracts" assignments from the PrimeNet server, and was
    previously the only means to obtain work at high bit levels. GIMPS now has a
    manual GPU assignment form that serves this purpose, but GPU to 72 remains
    a popular option.

    Please note results should be submitted to PrimeNet and not the GPU to 72
    website.

    GPU to 72 can be accessed here: https://gpu72.com

Using the MISFIT application:
    MISFIT is a Windows tool that automatically requests assignments and
    submits results. You can get it here: https://mersenneforum.org/misfit

    Important note: this program has reached end-of-life and is no longer
    supported. It is highly recommended to use AutoPrimeNet instead.

From mersenne.ca:
    James Heinrich's website mersenne.ca offers assignments for exponents up
    to 32 bits. You can get such work here: https://mersenne.ca/tf1G

    Be aware mfaktc currently does not support exponents below 100,000.

A note on extending assignments:
    Because modern GPUs are much more efficient than CPUs, they are often used
    to search for factors beyond traditional Prime95 limits:
    https://mersenne.org/various/math.php

    Users have historically edited worktodo.txt to manually extend assignments,
    but this is no longer necessary as both the manual GPU assignment form and
    GPU to 72 allow higher bit levels to be requested. However, the PrimeNet
    server still accepts results whose bit levels are higher than assigned.

    Please do not manually extend assignments from GPU to 72 as users are
    requested not to "trial factor past the level you've pledged."

---

    You must use mfaktc 0.24.0 or above starting in 2026 as the CRC32 checksums
    will be used to validate results.

    Once you have your assignments, create an empty file called worktodo.txt
    and copy all the "Factor=..." lines into that file. Start mfaktc, sit back
    and let it do its job. Running mfaktc is also a great way to stress test
    your GPU. ;-)

---

Submitting results:
    It is important to submit the results once you're done. Do not report
    partial results as PrimeNet may reassign the exponent to someone else in
    the meantime; this can lead to duplicate work and wasted cycles.

    AutoPrimeNet automatically submits results in addition to obtaining
    assignments. For computers without Internet access, you can manually submit
    the results instead:

    Step 1) log in to the GIMPS website with your username and password
    Step 2) on the menu bar, select Manual Testing > Results
    Step 3) upload the results.json.txt file produced by mfaktc. Do not submit
            the results.txt file as it is no longer accepted by the PrimeNet
            server. You may archive or delete the results.json.txt file after
            it has been processed.

    To prevent abuse, admin approval is required for manual submissions. You
    can request approval by contacting George Woltman at woltman@alum.mit.edu
    or posting on the GIMPS forum:
    https://mersenneforum.org/forumdisplay.php?f=38

##################
# 5 Known issues #
##################

- The user interface isn't hardened against malformed inputs. There are some
  checks, but when you try hard enough you should be able to screw it up.
- The GUI of your OS might be very laggy while running mfaktc. (newer GPUs
  with compute capability 2.0 or higher can handle this _MUCH_ better)
  Comment from James Heinrich:
    Slower/older GPUs (e.g. compute v1.1) that experience noticeable lag can
    get a significant boost in system usability by reducing the NumStreams
    setting from default "3" to "2", with minimal performance loss.
    Decreasing to "1" provides much greater system responsiveness, but also
    much lower throughput.
    At least it did so for me. With NumStreams=3, I could only run mfaktc
    when I wasn't using the computer. Now I run it all the time (except when
    watching a movie or playing a game...)
  Another thing worth trying is using different GridSize values in the INI
  file. Smaller grids should have higher responsiveness at the cost of slightly
  lower speed. Performance-wise, this is not recommended on GPUs which can
  handle more than 100 million candidates per second.
- the debug options CHECKS_MODBASECASE and USE_DEVICE_PRINTF might report 'qi'
  values that are too high while using the Barrett kernels; this is caused by
  factor candidates out of the specified range.


##################
# 5.1 Non-issues #
##################

- mfaktc runs slower on small ranges. Usually it doesn't make much sense to
  run mfaktc with an upper limit below 64 bits. mfaktc is designed to find
  factors between 64 and 92 bits, and is best suited for long-running jobs.

- mfaktc can find factors outside the given range.
  For example, './mfaktc -tf 66362159 40 41' has a high chance to report
  124246422648815633 as a factor. It is actually between 56 and 57 bits, so
  './mfaktc -tf 66362159 56 57' will find also this factor as usual.

  The reason for this behaviour is mfaktc works on huge factor blocks,
  controlled by GridSize in the INI file. The default value of GridSize=3 means
  mfaktc runs up to 1048576 factor candidates at once, per class. So the last
  block of each class is filled up with factor candidates to above the upper
  bit level. This is a huge overhead for small ranges but can be safely ignored
  for larger ranges. For example, the average overhead is 0.5% for a class with
  100 blocks but only 0.05% for one with 1000 blocks.


############
# 6 Tuning #
############

You can find additional settings in the mfaktc.ini file. Read it carefully
before making changes. ;-)


#########
# 7 FAQ #
#########

Q: Does mfaktc support multiple GPUs?
A: Currently no, but you can use the -d option to start an instance on a
   specific device. Please also see the next question.

Q: Can I run multiple mfaktc instances on the same computer?
A: Yes. In most cases, this is required to make full use of a GPU when sieving
   on the CPU. Otherwise, one instance should fully utilize a single GPU.

   You will need a separate directory for each mfaktc instance.

Q: Are checkpoint files compatible between different mfaktc versions?
A: Save files are compatible between different platforms and architectures. For
   example, the 32-bit Windows version can read a checkpoint from 64-bit Linux
   and vice versa.

   However, mfaktc 0.23.x and below can only load checkpoints with the same
   version number as the executable. Complete any active assignments before you
   upgrade.

Q: What do the version numbers mean?
A: mfaktc 0.23.0 and above use the semantic versioning scheme. You can learn
   more about semantic versioning here: https://semver.org

   You may come across pre-release versions that are not publicly available.
   Such versions are *not* intended for general use; sometimes they have the
   computational code disabled or don't even compile. Please don't use them for
   production work as they have usually had minimal to zero QA and may contain
   critical issues.

###########
# 8 .plan #
###########

0.24
- merge in changes from unreleased version 0.22
  - drop support for compute capability 1.x and 32-bit builds
  - CRC32 checksums to reduce invalid results
  - improved performance on Pascal devices
  - metadata in checkpoint file names
  - replace deprecated cudaThreadSynchronize() with cudaDeviceSynchronize()

ongoing improvements
- performance improvements whenever they are found ;-)
- fix bugs as they are discovered
- change compile-time options to runtime options, if applicable
- documentation and comments in code
- try to use double precision for the long integer divisions
  - unsure, may or may not be useful

requested features; no particular order and not planned for a specific release
- JSON output for Wagstaff numbers https://www.mersenneforum.org/showpost.php?p=662680&postcount=3769
- factors-meta.<factor>.timestamp https://www.mersenneforum.org/showpost.php?p=662603&postcount=3750
- factors-meta.<factor>.class https://www.mersenneforum.org/showpost.php?p=662720&postcount=3781
- found factors support https://www.mersenneforum.org/showpost.php?p=662682&postcount=3770
- begink and endk logging https://www.mersenneforum.org/showpost.php?p=662953&postcount=3845
- only log every n seconds https://www.mersenneforum.org/showpost.php?p=662795&postcount=3826
- catch HUP https://www.mersenneforum.org/showpost.php?p=662777&postcount=3815
- non-prime exponents https://www.mersenneforum.org/showpost.php?p=663442&postcount=3873
- TF10G support https://www.mersenneforum.org/showpost.php?p=663442&postcount=3873
- drop CPU sieving support https://www.mersenneforum.org/showpost.php?p=663517&postcount=3894
