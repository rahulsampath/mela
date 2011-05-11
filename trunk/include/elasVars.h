/*    int elasMultEvent;
    int hessMultEvent;
    int hessFinestMultEvent;
    int createHessContextEvent;
    int updateHessContextEvent;
    int evalObjEvent;
    int evalGradEvent;
    int createPatchesEvent;
    int optEvent;
    int computeSigEvent;
    int computeTauEvent;
    int computeGradTauEvent;
    int computeNodalTauEvent;
    int computeNodalGradTauEvent;
    int tauElemAtUEvent;
*/
#include <sys/time.h>

int elasMultEvent;
int hessMultEvent;
int hessFinestMultEvent;
int createHessContextEvent;
int updateHessContextEvent;
int computeSigEvent;
int computeTauEvent;
int computeGradTauEvent;
int computeNodalTauEvent;
int computeNodalGradTauEvent;
int evalObjEvent;
int evalGradEvent;
int createPatchesEvent;
int expandPatchesEvent;
int meshPatchesEvent;
int copyValsToPatchesEvent;
int optEvent;
int tauElemAtUEvent;

struct timeval  tp;
struct timezone tzp;

