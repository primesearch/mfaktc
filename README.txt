#################
# mfaktc README #
#################

Content

0   What is mfaktc?
1   Supported Hardware
2   Compilation
2.1 Compilation (Linux)
2.2 Compilation (Windows)
3   Running mfaktc (Linux)
3.1 Running mfaktc (Windows)
4   Getting work and reporting results
5   Known issues
5.1 Stuff that looks like an issue but actually isn't an issue
6   Tuning
7   FAQ
8   .plan


####################
# 0 What is mfaktc #
####################

mfaktc is a program for trial factoring of Mersenne numbers. The name mfaktc
is "Mersenne FAKTorisation with Cuda". Faktorisation is a mixture of the
English word "factorisation" and the German word "Faktorisierung".
It uses CPU and GPU resources.
It runs almost entirely on the GPU since v0.20 (previous versions used both
CPU and GPU resources).


########################
# 1 Supported Hardware #
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

It is assumed that you've already set up your compiler and CUDA environment.
There are some compiletime settings in the file src/params.h possible:
- in the upper part of the file there are some settings which "advanced
  users" can chance if they think it is beneficial. Those settings are
  verified for reasonable values.
- in the middle are some debug options which can be turned on. These options
  are only useful for debugging purposes.
- the third part contains some defines which should _NOT_ be changed unless
  you really know what they do. It is easily possible to screw up something.

A 64-bit build is preferred except for some old low-end GPUs because the
performance critical CPU code runs ~33% faster compared to 32-bit. (measured
on a Intel Core i7)


###########################
# 2.1 Compilation (Linux) #
###########################

Change into the subdirectory "src/"

Adjust the path to your CUDA installation in "Makefile" and run 'make'.
The binary "mfaktc" is placed into the parent directory.

I'm using
- OpenSUSE 12.2 x86_64
- gcc 4.7.1 (OpenSUSE 12.2)
- Nvidia driver 343.36
- Nvidia CUDA Toolkit
  - 6.5

Older CUDA Toolkit versions should work, too.

I didn't spend time testing mfaktc on 32bit Linux because I think 64bit
(x86_64) is adopted by most Linux users now. Anyway mfaktc should work on
32bit Linux, too. If problems are reported I'll try to fix them. So I don't
drop Linux 32bit support totally. ;)

When you compile mfaktc on a 32bit system you must change the library path
in "Makefile" (replace "lib64" with "lib").


#############################
# 2.2 Compilation (Windows) #
#############################

The following instructions have been tested on Windows 7 64bit using Visual
Studio 2012 Professional. A GNU compatible version of make is also required
as the Makefile is not compatible with nmake. GNU Make for Win32 can be
downloaded from http://gnuwin32.sourceforge.net/packages/make.htm.

Run the Visual Studio 2012 x64 Win64 Command Prompt for x64 or
Run the Visual Studio 2012 x86 Native Tools Command Prompt for x86 (32 bit)

 and change into the "\src" subdirectory.

Run 'make -f Makefile.win' for a 64bit built (recommended on 64bit systems)
or 'make -f Makefile.win32' for a 32bit built.

You will have to adjust the paths to your CUDA installation and the
Microsoft Visual Studio binaries in the makefiles if you have something
other than CUDA 6.5 and MSVS 2012. The binaries "mfaktc-win-64.exe" or
"mfaktc-win-32.exe" are placed in the parent directory.


############################
# 3 Running mfaktc (Linux) #
############################

Just run './mfaktc -h'. It will tell you what parameters it accepts.
Maybe you want to tweak the parameters in mfaktc.ini. A small description
of those parameters is included in mfaktc.ini, too.
Typically you want to get work from a worktodo file. You can specify the
name in mfaktc.ini. It was tested with PrimeNet v5 worktodo files but v4
should work, too.

Please run the builtin selftest each time you've
- recompiled the code
- downloaded a new binary from somewhere
- changed the Nvidia driver
- changed your hardware

Example worktodo.txt
-- cut here --
Factor=bla,66362159,64,68
Factor=bla,3321932839,50,71
-- cut here --

Than run e.g. './mfaktc'. If everything is working as expected this
should trial factor M66362159 from 2^64 to 2^68 and after that trial factor
M3321932839 from 2^50 to 2^71.


################################
# 3.1 Running mfaktc (Windows) #
################################

Similar to Linux (read above!).
Open a command window and run 'mfaktc.exe -h'.


########################################
# 4 Getting work and reporting results #
########################################

You must have a PrimeNet account to participate. Simply go to the GIMPS website
at https://mersenne.org and click "Register" to create one. Once you've signed
up, you can get assignments in several ways.

From the GIMPS website:
    Step 1) log in to the GIMPS website with your username and password
    Step 2) on the menu bar, select Manual Testing > Assignments
    Step 3) open the link to the manual GPU assignment request form
    Step 4) enter the number of assignments or GHz-days you want
    Step 5) click "Get Assignments"

    Users with older GPUs may want to use the regular form.

Using the GPU to 72 service:
    GPU to 72 "subcontracts" assignments from the PrimeNet server, and was
    previously the only means to obtain work at high bit levels. GIMPS now has a
    manual GPU assignment form that serves this purpose, but GPU to 72 remains
    a popular option.

    GPU to 72 website: https://gpu72.com

Using the MISFIT tool:
    MISFIT is a Windows tool that automatically requests assignments and
    submits results. You can get it here: https://mersenneforum.org/misfit

From mersenne.ca:
    James Heinrich's website mersenne.ca offers assignments for exponents up
    to 32 bits. You can get such work here: https://mersenne.ca/tf1G

    Be aware mfaktc currently does not support exponents below 100,000.

Advanced usage (extend the upper limit):
    Since mfaktc works best on long running jobs you may want to extend the
    upper TF limit of your assignments a little bit. Take a look how much TF
    is usually done here: http://www.mersenne.org/various/math.php
    Lets assume that you've received an assignment like this:
        Factor=<some hex key>,78467119,65,66
    This means that Primenet server assigned you to TF M78467119 from 2^65
    to 2^66. Take a look at the site noted above, those exponent should be
    TFed up to 2^71. Primenet will do this in multiple assignments (step by
    step) but since mfaktc runs very fast on modern GPUs you might want to
    TF up to 2^71 or even 2^72 directly. Just replace the 66 at the end of
    the line with e.g. 72 before you start mfaktc:
        e.g. Factor=<some hex key>,78467119,65,72

    It is important to submit the results once you're done. Do not report
    partial results as PrimeNet may reassign the exponent to someone else in
    the meantime; this can lead to duplicate work and wasted cycles. You can
    easily avoid this by setting Stages=0 in the INI file.

    Please do not manually extend assignments from GPU to 72 as users are
    requested not to "trial factor past the level you've pledged."


    Once you have your assignments, create an empty file called worktodo.txt
    and copy all the "Factor=..." lines into that file. Start mfaktc, sit back
    and let it do its job. Running mfaktc is also a great way to stress test
    your GPU. ;-)

Submitting results:
    mfaktc does not natively communicate with the PrimeNet server, but there is
    a program called AutoPrimeNet that does so. See the AutoPrimeNet download
    page for instructions: https://download.mersenne.ca/AutoPrimeNet

    Another option is to manually submit the results:

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
  with compute capabilty 2.0 or higher can handle this _MUCH_ better)
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
  it's size is between 2^56 and 2^57! Of course
  './mfaktc -tf 66362159 56 57' will find this factor, too. The reason
  for this behaviour is that mfaktc works on huge factor blocks. This is
  controlled by GridSize in mfaktc.ini. The default value is 3 which means
  that mfaktc runs up to 1048576 factor candidates at once (per class). So
  the last block of each class is filled up with factor candidates above to
  upper limit. While this is a huge overhead for small ranges it's save to
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