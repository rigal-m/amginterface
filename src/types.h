#ifndef TYPES_HEADER__
#define TYPES_HEADER__

/* #include <fefloa_include.h> */

#ifdef i4 
 #define mint int
 #ifndef _type_int1_ 
   #define _type_int1_ 
   #define int1 int 
 #endif
#else 
  #ifdef  i8 
    #define mint long long int 
    #ifndef _type_int1_ 
    #define _type_int1_ 
      #define int1 long long int 
    #endif
  #endif   
#endif

#define FILE_SU2    1
#define FILE_GMF    2
#define FILE_DAT    3
#define FILE_GMFSOL 4
#define FILE_GEO    5
#define FILE_MSH    6

#define SU2_LINE         3
#define SU2_TRIANGLE     5
#define SU2_RECTANGLE    9
#define SU2_TETRAHEDRAL  10
#define SU2_HEXAHEDRAL   12
#define SU2_WEDGE        13
#define SU2_PYRAMID      14
#define SU2_TRIANGLEP2   105
#define SU2_LINEP2       103

#define GMSH_EDGE         1
#define GMSH_TRIANGLE     2
#define GMSH_HEXAHEDRON   3
#define GMSH_TETRAHEDRON  4

typedef double double2[2];
typedef double double3[3];
typedef int    int2[2];
typedef int    int3[3];
typedef int    int4[4];
typedef int    int5[5];
typedef int    int6[6];
typedef int    int7[7];
typedef int    int8[8];
typedef int    int9[9];

typedef char byte_t;

/* Minimal FefloaPython Mesh */
typedef struct FefloaPythonMesh
{
    int Dim, NbrVer, NbrTri, NbrTet, NbrEdg, NbrCor, NbrReq;
  
    double3 *crd;
    mint    *refver;
  
    int4 *bface;
    mint *lfasu;
  
    int4 *intmat;
    mint *ledom;
  
    int3 *edg;
  
    mint *cor;
  
} FefloaPythonMesh;

typedef struct S_Mesh
{
    int	NbrVer;   	/* number of vertices  			  */
    int	NbrTri;   	/* number of triangles 			  */
    int	NbrEfr;     /* number of boundary edges   */
    int	NbrTet;     
    int	NbrQua;
    int NbrHex;
    int NbrPri;
    int NbrPyr;
	
    int	MaxNbrVer;   	/* number of vertices  			  */
    int	MaxNbrTri;   	/* number of triangles 			  */
    int	MaxNbrEfr;     /* number of boundary edges   */
    int	MaxNbrTet;
    int	MaxNbrQua;
    int MaxNbrHex;
    int MaxNbrPri;
    int MaxNbrPyr;
	
    double3    *Ver;	    
    int4       *Tri;	    
    int3       *Efr;   
    int5       *Tet;
    int9       *Hex;
    int7       *Pri;
    int5       *Qua;	
    int6       *Pyr;
	
    int         NbrMarkers;
    char       	Markers[10000][1024];
	
    int         SolSiz;   /* Solution size */
    double     *Sol;
    int         NbrFld;   /* Number of solution fields */
    int        *FldTab;   /* Type of each field (scalar, vector, etc.) as defined by libmesh6*/
    char       SolTag[100][256];
	
    int Dim;
	
    char    MshNam[1024];
    char    SolNam[1024];
	
    int FilTyp;
	  
} Mesh;

typedef struct T_Options
{
	
    int  Mod;              // module
	
    int  InpFilTyp; // input file's type: .mesh, .su2, etc.
	
    char InpNam[1024];   // Input mesh file name
    char BasNam[1024];   // Base name from InpNam
    char OutNam[1024];   // Output file name
    char SolNam[1024];   // Input solution name
	
    char HeaderNam[1024]; // SU2 solution file name used to copy the header information
	
    int flagSol;  // Output solution?
	
    int clean; // clean mesh?
    int Dim; // force dim to be 2?
	
    double Box[6];
	
} Options;

#endif // TYPES HEADER
