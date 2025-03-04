#################
# mfaktc README #
#################

Content

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
5.1 Stuff that looks like an issue but actually isn't an issue
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

Be aware that 32-bit applications are not supported in CUDA Toolkit 12.2 and
later. You will need to use an older CUDA Toolkit to build mfaktc for 32 bits.
See this thread for details:
https://forums.developer.nvidia.com/t/whats-the-last-version-of-the-cuda-toolkit-to-support-32-bit-applications/323106/4

In any case, a 64-bit build is preferred except on some old low-end GPUs.
Testing on an Intel Core i7 CPU has shown that the performance-critical CPU
code runs about 33% faster compared to 32 bits.

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

nvcc may exit due to an "unsupported gpu architecture" error. If this happens,
simply comment out the corresponding "NVCCFLAGS += ..." line in the makefile.
You may have to do this more than once. Otherwise, the binary "mfaktc" should
appear in the parent folder.

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
http://gnuwin32.sourceforge.net/packages/make.htm

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

mfaktc works very similarly on Windows. See the above instructions, but run
"mfaktc" without the "./" to launch the executable.


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
    Step 3) upload the results.json.txt file produced by mfaktc. You may
            archive or delete the file after it has been processed.

    To prevent abuse, admin approval is required for manual submissions. You
    can request approval by contacting George Woltman at woltman@alum.mit.edu
    or posting on the GIMPS forum:
    https://mersenneforum.org/forumdisplay.php?f=38

    Important note: the results.txt file is deprecated and will no longer be
    accepted from 2025 onwards.

##################
# 5 Known issues #
##################

- The user interface isn't hardened against malformed input. There are some
  checks but when you really try you should be able to screw it up.
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
  Another thing worth trying is different settings of GridSize in
  mfaktc.ini. Smaller grids should have higher responsibility with the cost
  of a little performance penalty. Performancewise this is not recommended
  on GPUs which can handle >= 100M/s candidates.
- the debug options CHECKS_MODBASECASE (and USE_DEVICE_PRINTF) might report
  too high qi values while using the barrett kernels. They are caused by
  factor candidates out of the specified range.


##################################################################
# 5.1 Stuff that looks like an issue but actually isn't an issue #
##################################################################

- mfaktc runs slower on small ranges. Usually it doesn't make much sense to
  run mfaktc with an upper limit smaller than 2^64. It is designed for trial
  factoring above 2^64 up to 2^95 (factor sizes). ==> mfaktc needs
  "long runs"!
- mfaktc can find factors outside the given range.
  E.g. './mfaktc -tf 66362159 40 41' has a high change to report
  124246422648815633 as a factor. Actually this is a factor of M66362159 but
  its size is between 2^56 and 2^57! Of course
  './mfaktc -tf 66362159 56 57' will find this factor, too. The reason
  for this behaviour is that mfaktc works on huge factor blocks. This is
  controlled by GridSize in mfaktc.ini. The default value is 3 which means
  that mfaktc runs up to 1048576 factor candidates at once (per class). So
  the last block of each class is filled up with factor candidates above to
  upper limit. While this is a huge overhead for small ranges it's safe to
  ignore it on bigger ranges. If a class contains 100 blocks the overhead is
  on average 0.5%. When a class needs 1000 blocks the overhead is 0.05%...


############
# 6 Tuning #
############

Read mfaktc.ini and read before editing. ;)


#########
# 7 FAQ #
#########

Q Does mfaktc support multiple GPUs?
A Yes, with the exception that a single instance of mfaktc can only use one
  GPU. For each GPU you want to run mfaktc on you need (at least) one
  instance of mfaktc. For each instance of mfaktc you can use the
  commandline option "-d <GPU number>" to specify which GPU to use for each
  specific mfaktc instance. Please read the next question, too.

Q Can I run multiple instances of mfaktc on the same computer?
A Yes! You need a separate directory for each instance of mfaktc.

Q Can I continue (load a checkpoint) from a 32bit version of mfaktc with a
  64bit version of mfaktc (and vice versa)?
A Yes!

Q Version numbers
A release versions are usually 0.XX where XX increases by one for each new
  release. Sometimes there are version which include a single (quick) patch.
  If you look into the Changelog.txt you can see the mfaktc 0.13 was
  followed by mfaktc 0.13p1 followed by mfaktc 0.14. These 0.XXpY versions
  are intended for daily work by regular users!
  Additionally there are lots of 0.XX-preY versions which are usually not
  public available. They are usually *NOT* intended for productive usage,
  sometimes they don't even compile or have the computational part disabled.
  If you somehow receive one of those -pre versions please don't use them
  for productive work. They had usually minimal to zero QA.


###########
# 8 .plan #
###########

0.22
- merge "worktodo.add" from mfakto <-- done in 0.21
- check/validate mfaktc for lower exponents <-- done in 0.21
- rework debug code
- fast (GPU-sieve enabled) kernel for factors < 2^64?

0.??
- automatic primenet interaction (Eric Christenson is working on this)         <- specification draft exists; on hold, Eric doesn't want to continue his efforts. :(
  - this will greatly increase usability of mfaktc
  - George Woltman agreed to include the so called "security module" in
    mfaktc for a closed source version of mfaktc. I have to check license
    options, GPL v3 does not allow to have parts of the program to be
    closed source. Solution: I'll re-release under another license. This is
    NOT the end of the GPL v3 version! I'll release future versions of
    mfaktc under GPL v3! I want mfaktc being open source! The only
    differences of the closed version will be the security module and the
    license information.

not planned for a specific release yet, no particular order!
- performance improvements whenever I find them ;)
- change compiletime options to runtime options (if feasible and useful)
- documentation and comments in code
- try to use double precision for the long integer divisions                  <-- unsure
- json output for wagstaff numbers https://www.mersenneforum.org/showpost.php?p=662680&postcount=3769
- factors-meta.<factor>.timestamp https://www.mersenneforum.org/showpost.php?p=662603&postcount=3750
- factors-meta.<factor>.class https://www.mersenneforum.org/showpost.php?p=662720&postcount=3781
- found factors support https://www.mersenneforum.org/showpost.php?p=662682&postcount=3770
- os info https://www.mersenneforum.org/showpost.php?p=662648&postcount=3757
- security checksum https://www.mersenneforum.org/showpost.php?p=662658&postcount=3761
- detailed runtime logging https://www.mersenneforum.org/showpost.php?p=662953&postcount=3845
- begink and endk logging https://www.mersenneforum.org/showpost.php?p=662953&postcount=3845
- only log every n seconds https://www.mersenneforum.org/showpost.php?p=662795&postcount=3826
- catch HUP https://www.mersenneforum.org/showpost.php?p=662777&postcount=3815
- non-prime exponents https://www.mersenneforum.org/showpost.php?p=663442&postcount=3873
- TF10G support https://www.mersenneforum.org/showpost.php?p=663442&postcount=3873
- Remove CPU Sieving support https://www.mersenneforum.org/showpost.php?p=663517&postcount=3894