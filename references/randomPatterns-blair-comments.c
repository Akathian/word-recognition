#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define TRUE  1
#define FALSE 0

#define MAX_NMEMBERS	3000
#define MAX_NFEATURES	100

char members[MAX_NMEMBERS][MAX_NFEATURES] ;
//item is defined here outside of all the other loops... why?  And it is defined as size of the maxiumu number
//of features, instead of the actual number used in the program...
char item[MAX_NFEATURES] ;


int	flip(prob)
  float prob ;
{
  //drand is just a random number generator in c, returning uniform distribution of numbers between 0 and 1
  if (drand48() < prob) return TRUE ;
  else                  return FALSE ;
}

//argc is the size of the array, whichc in c I believe needs to be passed around everywhere... 
//the language is not smart enough to remember array sizes
main (argc, argv)
int argc ;
char *argv[] ;
{
  int c, oc, m, om, f, n, seed ;
  int minDiff, minOn, maxOn, new, nDiff, nOn, nMembers, nFeatures ;
  float probOn ;

  if (argc != 8) {
	//here argv refers to the actual name of the program that was given on the command line
    printf("Usage: %s <nFeatures> <nMembers> <probOn> <minOn> <maxOn> <minDiff> <seed>\n", argv[0]) ;
    exit(0) ;
  }
  nFeatures	= atoi(argv[1]) ;
  nMembers	= atoi(argv[2]) ;
  probOn	= atof(argv[3]) ;
  minOn		= atoi(argv[4]) ;
  maxOn		= atoi(argv[5]) ;
  minDiff	= atoi(argv[6]) ;
  seed		= atoi(argv[7]) ;

  srand48(seed) ;
  
//m is the counter for now many members there are
m = 0 ;
  while (m < nMembers) {
    /* 
     * generate new potential item
     */
    new = TRUE ;
    nOn = 0 ;
    for (f = 0 ; f < nFeatures ; f++) {
      if (flip(probOn)) {
	item[f] = 1 ;
	nOn++ ;
      } else 
	item[f] = 0 ;
    }
	// in this case, the continue goes back to the start of the while statement, thus restarting the generation
	// of this particular vector
    if (nOn < minOn || nOn > maxOn) continue ;
    /*
     * test against existing items
     */
    for (om = 0 ; om < m && new ; om++) {
      nDiff = 0 ;
      for (f = 0 ; f < nFeatures ; f++) 
	if (item[f] != members[om][f]) nDiff++ ;
      if (nDiff < minDiff) new = FALSE ;
    }
    if (!new) continue ;
    /*
     * save new item
     */
    for (f = 0 ; f < nFeatures ; f++) 
      members[m][f] = item[f] ;
    m++ ;
  }
  /*
   * print items
   */
  for (m = 0 ; m < nMembers ; m++) {
    for (f = 0 ; f < nFeatures ; f++) printf("%1d ",members[m][f]) ;
    printf("\n") ;
  }
}
