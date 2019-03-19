# asteroids
... on a torus, Klein bottle, or projective plane.

---

This version of the classic game, Asteroids, was largely live-coded for a game design course taught in Shanghai in Spring
2015.  The students in that course had varying degrees of coding experience.  The game runs in Python3 using
[pygame](https://www.pygame.org/wiki/GettingStarted).

Notes:
* The Intro screen describes the controls.
* To get a feel for the game, you can fly around and fire bullets while you're on the Intro screen.  You can also
  switch the topological space on which you are playing.
* Undocumented feature: if you hit the down arrow, you're ship brakes immediately.  This only works of the Intro screen.
  Try moving onto an edge or a corner in each the three different spaces.
* For an added twist to the game play, if you are playing on a Klein bottle or projective space, whenever you pass
  an edge that was glued to its opposite with a half-twist, the ship gets reflected so that your burners switch sides.
* Some students were concerned about wearing out their space bar.  Hit 'b' for a burst of bullets.
* Obviously the game isn't finished.  Issue a pull request if you work on it!

---

Detailed instructions for get this running on Windows:

Use Chrome as your browser instead of Microsoft Edge. Go here: https://www.google.com/chrome/ and install it, if it is not already installed.

To install Python, go here:  https://www.python.org/, mouse over Downloads, and then click on the python-3.7.2.exe button.

That should have download a file called python-3.7.2.exe that should occur in a tab at the bottom of the Chrome browser.   Click on that tab.  That should start the installer. 

When the installer window opens, click the box at the bottom that says Add Python 3.7 to Path.  Then click Install Now.   

--------------
Now Python should be installed 
--------------

After you have Python3 installed, you get that Asteroids game from my Github site running as follows:

First, install PyGame by doing this:
In the search box at the lower left of the Windows desktop, type Command Prompt, and hit return. 
At the command prompt issue the command:  py -m pip install -U pygame --user
To see if things are working, issue this at the command line:  py -m pygame.examples.aliens
A rudimentary game should have popped up.
      ------------------
      Now PyGame is installed
      ------------------

Next, install Git as follows:
Go here:  https://git-scm.com/download/win  in Chrome
That should have automatically downloaded an installer.  Click on its filename in the bottom left of Chrome.  Accept all the default configurations.
       ----------------
       Now Git is installed 
       ----------------

Next, use git to download that Asteroids game from its repository on my Github page:
Just to make sure system paths have been updated, close out the command line session from before and start up a new one.
Issue the command:   git clone https://github.com/sj-simmons/asteroids
       ----------------
       Now a subdirectory in your home directory call asteroids should exist that contains the files from my repo 
       ----------------
Check and see if the directory is there by issuing the command:  dir
If it's there, then change into the directory by issuing:  cd asteroids
Now use the dir command to see what's in the asteroids directory. 
There should be like 5 files in there, the largest being asteroids.py
Now run the game by issuing the command:  python asteroids.py
On the CONTROLS screen at the beginning of the game, you can switch topological spaces and fly around and even shoot, before you start playing.

---

### Detailed notes for installing this on Windows

#### First install Python3:

* Use Chrome as your browser instead of Microsoft Edge. Go here:
  [google.com/chrome/](https://www.google.com/chrome/) and install it, if it is not already installed.
* To install Python, go to [python.org](https://www.python.org/), mouse over *Downloads*, and then
  click on the *python-3.7.2.exe* button (or the whatever the latest version number is).
* The last step should have downloaded a file called *python-3.7.2.exe* that should show in a tab at
  the bottom of your Chrome browser.  Click on that tab. The installer should start.
* When the installer window opens, click the box at the bottom that says *Add Python 3.7 to Path*.
  Then click *Install Now*.
* Python should now be installed!

#### After you have Python3 installed, install PyGame:
* In the search box at the lower left of the Windows desktop, type _Command Prompt_, and hit the *return* key.
* At the command prompt, issue the command:  `py -m pip install -U pygame --user`
* To see if things are working correctly, issue this at the command line:  `py -m pygame.examples.aliens`
* A rudimentary game should have popped up.
* Now PyGame is installed!

#### Next, install Git as follows:
* Go to [git-scm.com/download/win](https://git-scm.com/download/win) in Chrome.
* That should have automatically downloaded an installer.  Click on its filename in the bottom left of Chrome.
  Accept all the default configurations.
* Now Git is installed!

#### Now, let's use git to download the Asteroids game from its repository on my Github page:
* Just to make sure system paths have been updated, close out the command line session from before and start up a new one.
* Issue the command:   `git clone https://github.com/sj-simmons/asteroids`
* Now a subdirectory in your home directory call asteroids should exist that contains the files from this repo!
* Check and see if the directory is there by issuing the command: `dir`.
* If the *asteroids* directory is listed, then change into that directory by issuing the command:  `cd asteroids`.
* Now use the `dir` command to see what's in the asteroids directory.
* There should be like 5 files in there, the largest being _asteroids.py_.
* Now run the game by issuing the command:  `python asteroids.py`
* On the CONTROLS screen at the beginning of the game, you can switch topological spaces, fly around, and even shoot,
  before you start playing.



