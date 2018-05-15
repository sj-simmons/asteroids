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
  Try moving onto an edge or a corner in the three different spaces.
* For an added twist to the game play, if you are playing on a Klein bottle or projective space, whenever you pass
  an edge that was glued to its opposite with a half-twist, the ship gets reflected so that your burners switch sides.
* Some students were concerned about wearing out their space bar.  Hit 'b' for a burst of bullets.
* Obviously the game isn't finished.  Issue a pull request if you work on it!
