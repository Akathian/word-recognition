#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define TRUE  1
#define FALSE 0

#define MAX_NMEMBERS	3000
#define MAX_NFEATURES	100

char members[MAX_NMEMBERS][MAX_NFEATURES] ;
char item[MAX_NFEATURES] ;


int	flip(prob)
  float prob ;
{
  if (drand48() < prob) return TRUE ;
  else                  return FALSE ;
}


main (argc, argv)
int argc ;
char *argv[] ;
{
  int c, oc, m, om, f, n, seed ;
  int minDiff, minOn, maxOn, new, nDiff, nOn, nMembers, nFeatures ;
  float probOn ;

  if (argc != 8) {
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