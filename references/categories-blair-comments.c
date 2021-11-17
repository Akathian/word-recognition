#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define TRUE  1
#define FALSE 0

#define MAX_NCATEGORIES	150
#define MAX_NMEMBERS	100
#define MAX_NFEATURES	2000

char cats[MAX_NCATEGORIES][MAX_NMEMBERS][MAX_NFEATURES] ;
char proto[MAX_NFEATURES] ;
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
  int c, oc, m, om, f, n ;
  int minDiff, new, nDiff, nOn, seed, sparse ;
  int	nFeatures, nCategories, nMembers ;
  float minProbOn, maxProbOn, probOn, minProbDistort, maxProbDistort, probDistort ;

  if (argc != 11) {
    printf("Usage: %s <nFeatures> <nCategories> <nMembers> <minProbOn> <maxProbOn> <minDiff> <minProbDistort> <maxProbDistort> <sparseFormat> <seed>\n", argv[0]) ;
    exit(0) ;
  }
  nFeatures	= atoi(argv[1]) ;	/* number of features per pattern */
  nCategories	= atoi(argv[2]) ;	/* number of clusters (prototypes) */
  nMembers	= atoi(argv[3]) ;	/* number of exemplars per cluster */
  minProbOn	= atof(argv[4]) ;	/* maximum sparcity of prototype */
  maxProbOn	= atof(argv[5]) ;	/* minimum sparcity of prototype */
  minDiff	= atoi(argv[6]) ;	/* minimum bit-wise difference among exemplars */
  minProbDistort= atof(argv[7]) ;	/* min prob that feature is regenerated */
  maxProbDistort= atof(argv[8]) ;	/* max prob that feature is regenerated */
  sparse	= atoi(argv[9]) ;	/* generate output in "sparse" (unit numbers) format */
  seed		= atoi(argv[10]) ;	/* random number seed */

  if (nFeatures > MAX_NFEATURES || nCategories > MAX_NCATEGORIES || nMembers > MAX_NMEMBERS) {
    printf("nFeatures %d > MAX_NFEATURES %d | nCategories %d > MAX_NCATEGORIES %d | nMembers %d > MAX_NMEMBERS %d",
      nFeatures, MAX_NFEATURES, nCategories, MAX_NCATEGORIES, nMembers, MAX_NMEMBERS) ;
    exit(0) ;
  }
  printf("# %s %d %d %d %f %f %d %f %f %d\n", 
	 argv[0],nFeatures,nCategories,nMembers,minProbOn,maxProbOn,minDiff,minProbDistort,minProbDistort,seed) ;

  srand48(seed) ;
  for (c = 0 ; c < nCategories ; c++) {
//******* WHY????
//why does he change the probOn between these two levels half way through?  Maybe Plaut had some original reason for this
    probDistort = minProbDistort ;
    if (c < nCategories/2) probOn = minProbOn ;
    else		   probOn = maxProbOn ;
    /*
    if (c < nCategories/2) probDistort = minProbDistort ;
    else		   probDistort = maxProbDistort ;
    probOn = minProbOn + c*(maxProbOn-minProbOn)/((float)(nCategories-1)) ;
    */
    /* 
     * generate new prototype 
     * (with exact correct number of ON features)
     */
    for (f = 0 ; f < nFeatures ; f++) proto[f] = 0 ;
    nOn = (int)(0.5+probOn*nFeatures) ;
//this is the part that is the prototype which is being formed...
    for (n = 0 ; n < nOn ; ) {
	
	//select a random feature that will have its state turned on as part of the prototype   
   f = (int)(drand48()*nFeatures) ;
      if (proto[f] == 0) {
	proto[f] = 1 ;
	//only increment n if a new unit was successfully changed
	n++ ;
      }
    }

    m = 0 ;
    while (m < nMembers) {
      /* 
       * generate new potential item
       */

      for (f = 0 ; f < nFeatures ; f++) 
   //if the first expression is true (flip(probDistrot) then it runs the first statement, if it is false, it runs the second
  //in this case, it first evaluates whether the item should be distored.  If yes, it then flip's its state with the prob
//that a unit is on...  if it shouldn't be distored, it keeps the same value.
	item[f] = (flip(probDistort) ? flip(probOn) : proto[f]) ;
      /*
       * test against existing items in same category
       */
      for (om = 0 ; om < m && new ; om++) {
	nDiff = 0 ;
	for (f = 0 ; f < nFeatures ; f++) 
	  if (item[f] != cats[c][om][f]) nDiff++ ;
	if (nDiff < minDiff) new = FALSE ;
      }
      if (!new) continue ;
      /*
       * test against existing items in other categories
       * (do this first because it's most likely to fail)
       */
      for (oc = 0 ; oc < c && new ; oc++) {
	for (om = 0 ; om < nMembers && new ; om++) {
	  nDiff = 0 ;
	  for (f = 0 ; f < nFeatures ; f++) 
	    if (item[f] != cats[oc][om][f]) nDiff++ ;
	  if (nDiff < minDiff) new = FALSE ;
	}
      }
      if (!new) continue ;
      /*
       * save new item
       */
      for (f = 0 ; f < nFeatures ; f++) 
	cats[c][m][f] = item[f] ;
      m++ ;
    }
  /*
   * print items
   */
    if (sparse) {
      for (m = 0 ; m < nMembers ; m++) {
	for (f = 0 ; f < nFeatures ; f++) if (cats[c][m][f]) printf("%d ", f) ;
	printf("\n") ;
      }
    } else {
      printf("CATEGORY %d: probOn = %f\n", c, probOn) ;
      printf("Prototype: nOn = %d\n", nOn) ;
      for (f = 0 ; f < nFeatures ; f++) printf("%1d",proto[f]) ;
      printf("\nMembers:\n") ;
      for (m = 0 ; m < nMembers ; m++) {
	for (f = 0 ; f < nFeatures ; f++) printf("%1d ",cats[c][m][f]) ;
	printf("\n") ;
      }
    }
    fflush(stdout) ;
  }
}
