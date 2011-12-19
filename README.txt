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
4   Howto get work and report results from/to the primenet server
5   Known issues
5.1 Stuff that looks like an issue but actually isn't an issue
6   Tuning
7   FAQ
8   .plan



####################
# 0 What is mfaktc #
####################

mfaktc is a program for trial factoring of mersenne numbers. The name mfaktc
is "Mersenne FAKTorisation with Cuda". Faktorisation is a mixture of the
english word "factorisation" and the german word "Faktorisierung".
It uses CPU and GPU resources.



########################
# 1 Supported Hardware #
########################

mfaktc should run all all CUDA capable Nvidia GPUs with compute capability
>= 1.1. From my knowledge there is only once CUDA capable GPU with compute
capability 1.0: the G80 chip which is found on Geforce 8800 Ultra / GTX /
GTS 640 / GTS 320 and their Quadro and Tesla variants.



#################
# 2 Compilation #
#################

It is assumed that you've allready setup your compiler and CUDA environment.
There are some compiletime settings in the file src/params.h possible:
- in the upper part of the file there are some settings which "advanced
  users" can chance if they think it is beneficial. Those settings are
  verified for reasonable values.
- in the middle are some debug options which can be turned on. These options
  are only usefull for debuging purposes.
- the third part contains some defines which should _NOT_ be changed unless
  you really know what they do. It is easily possible to screw up something.

A 64bit built is prefered except for some old lowend GPUs because the
performance critical CPU code runs ~33% faster compared to 32bit. (measured
on a Intel Core i7)



###########################
# 2.1 Compilation (Linux) #
###########################

Change into the subdirectory "src/"

Adjust the path to your CUDA installation in "Makefile" and run 'make'.
The binary "mfaktc.exe" is placed into the parent directory.

I'm using
- OpenSUSE 11.1 x86_64
- gcc 4.3.2 (OpenSUSE 11.1)
- Nvidia driver 260.24
- Nvidia CUDA Toolkit
  - 3.2       ~1% slower than 3.0/3.1
  - 3.1       read below
  - 3.0       read below

CUDA Toolkits 3.0 and 3.1 are only minimal tested (compile, run selftest).

I don't spent time testing mfaktc on 32bit Linux because I think 64bit
(x86_64) is adopted by most Linux users now. Anyway mfaktc should work on
32bit Linux, too. If problems are reported I'll try to fix them. So I don't
drop Linux 32bit support totally. ;)

When you compile mfaktc on a 32bit system you must change the library path
in "Makefile" (replace "lib64" with "lib").



#############################
# 2.2 Compilation (Windows) #
#############################

The following instructions have been tested on Windows 7 64bit using Visual
Studio 2008 Professional. A GNU compatible version of make is also required
as the Makefile is not compatible with nmake. GNU Make for Win32 can be
downloaded from http://gnuwin32.sourceforge.net/packages/make.htm.

Run the Visual Studio 2008 x64 Win64 Command Prompt and change into the
"src/" subdirectory.

Run 'make -f Makefile.win' for a 64bit built (recommended on 64bit systems)
or 'make -f Makefile.win32' for a 32bit built. Perhaps you have to adjust
the paths to your CUDA installation and the Microsoft Visual Studio binaries
in the makefiles. The binaries "mfaktc-win-64.exe" or "mfaktc-win-32.exe"
are placed in the parent directory.



############################
# 3 Running mfaktc (Linux) #
############################

Just run './mfaktc.exe -h'. It will tell you what parameters it accepts.
Maybe you want to tweak the parameters in mfaktc.ini. A small describtion
of those parameters is included in mfaktc.ini, too.
Typically you want to get work from a worktodo file. You can specify the
name in mfaktc.ini. It was tested with primenet v5 worktodo files but v4
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

Than run e.g. './mfaktc.exe'. If everything is working as expected this
should trial factor M66362159 from 2^64 to 2^68 and after that trial factor
M3321932839 from 2^50 to 2^71.



################################
# 3.1 Running mfaktc (Windows) #
################################

Similar to Linux (read above!).
Open a command shell and run 'mfaktc.exe -h'.



###################################################################
# 4 Howto get work and report results from/to the primenet server #
###################################################################

Getting work:
    Step 1) go to http://www.mersenne.org/ and login with your username and
            password
    Step 2) on the menu on the left click "Manual Testing" and than
            "Assignments"
    Step 3) choose the number of assignments by choosing
            "Number of CPUs (cores) you need assignments for (maximum 12)"
            and "Number of assignments you want for each core"
    Step 4) Change "Preferred work type" to "Trial factoring"
    Step 5) click the button "Get Assignments"
    Step 6) copy&paste the "Factor=..." lines directly into the worktodo.txt
            in your mfaktc directory

Start mfaktc and stress your GPU ;)

Once mfaktc has finished all the work report the results to the primenet
server:
    Step 1) go to http://www.mersenne.org/ and login with your username and
            password
    Step 2) on the menu on the left click "Manual Testing" and than
            "Results"
    Step 3) upload the results.txt file generated by mfaktc using the
            "search" and "upload" button
    Step 49 once you've verified that the primenet server has recognized
            your results delete or rename the results.txt from mfaktc

Advanced usage (extend the upper limit):
    Since mfaktc works best on long running jobs you may want to extend the
    upper TF limit of your assignments a little bit. Take a look how much TF
    is usually done here: http://www.mersenne.org/various/math.php
    Lets assume that you've received an assignment like this:
        Factor=<some hex key>,78467119,65,66
    This means that primenet server assigned you to TF M78467119 from 2^65
    to 2^66. Take a look at the site noted above, those exponent should be
    TFed up to 2^71. Primenet will do this in multiple assignments (step by
    step) but since mfaktc runs very fast on modern GPUs you might want to
    TF up to 2^71 or even 2^72 directly. Just replace the 66 at the end of
    the line with e.g. 72 before you start mfaktc:
        e.g. Factor=<some hex key>,78467119,65,72
    When you increase the upper limit of your assignments it is import to
    report the results once you've finished up to the desired level. (Do not
    report partially results before!)



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
  Another try worth are different settings of GridSize in mfaktc.ini.
  Smaller grids should have higher responsibility with the cost of a little
  performance penalty. Performancewise this is not recommended on GPUs which
  can handle >= 100M/s candidates.
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
  E.g. './mfaktc.exe -tf 66362159 40 41' has a high change to report
  124246422648815633 as a factor. Actually this is a factor of M66362159 but
  it's size is between 2^56 and 2^57! Offcourse
  './mfaktc.exe -tf 66362159 56 57' will find this factor, too. The reason
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

Read mfaktc.ini and think before edit. ;)



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

0.19
- automatic primenet interaction (Eric Christenson is working on this)         <- specification draft exists
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
- try to use double precission for the long integer divisions                  <-- unsure
