/*
	Copyright (C) 2015  Zhong Wang

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


/**
	\file svmlight.cpp
	\brief R interface of svmlight
*/

#include "headers.hpp"
#include <R.h>
#include <Rinternals.h>
#include <Rembedded.h>
#include <Rdefines.h>

class optimVector{

public:
	optimVector( size_t nReserv, float fIncRate )
	{
        m_nActLen = 0;
        m_fIncRate = fIncRate;
		m_nSize = nReserv;
        m_pBuffer = Calloc( nReserv + 1, float);

	}
    virtual ~optimVector()
    {
		Free(m_pBuffer);
	}


    inline size_t size()
    {
        return(m_nActLen);
    }

    inline void push_back(float val)
    {
		if(m_nActLen>=m_nSize)
		{
			size_t newsize = round(m_nSize *( 1 + m_fIncRate) );
			float* pNew = Calloc( newsize, float);
			memcpy(pNew,  m_pBuffer, m_nSize*sizeof(float));
			Free(m_pBuffer);
			m_pBuffer = pNew;
			m_nSize = newsize;
		}

        m_pBuffer[m_nActLen] = val;
        m_nActLen++;
    }

    inline float operator[](unsigned int row) const
    {
        if (row>=m_nActLen || row<0)
           throw("wrong index of vector");
        return m_pBuffer[row] ;
    };

public:
   size_t m_nSize;
   size_t m_nActLen;
   float m_fIncRate;
   float* m_pBuffer;
};

inline char* getline( char* buffer, unsigned int* pOffset, bool isEof )
{
	char* pstart = buffer + *pOffset;
	char* pos = (char*)strstr(pstart, "\n");
	if( pos!=NULL )
	{
		*pos = '\0';

		if ( (pos-1) > pstart && *(pos-1) =='\r' )
		    *(pos-1) = '\0';

		if ( *(pos+1) =='\r' )
		{
			pos++;
			*pos = '\0';
	    }

		*pOffset = ( pos + 1 ) - buffer;
		return(pstart);
	}

	if(isEof)
	{
		*pOffset += strlen(pstart);
		return(pstart);
	}

	return(NULL);
}

extern "C" SEXP get_svmlight( SEXP filename )
{
	unsigned int rows    = 0;
	unsigned int columns = 0;
	unsigned int offset  = 0;

	std::ifstream file( CHAR(STRING_ELT(filename, 0)) );
	if ( file.fail() )
	{
		Rprintf("Error: Unable to open dataset file");
		return(R_NilValue);
	}

    file.seekg(0, file.end);
    int length = file.tellg();
    if ( length <= 0 )
        length = 1000*1000*1000;
    size_t nEstLen = round(length/10);

//Rprintf( "File length=%d vector size=%d\n", length, nEstLen );
	optimVector values(nEstLen,  0.5);
	optimVector labels(nEstLen,  0.5);
	optimVector lines(nEstLen,   0.5);
	optimVector indices(nEstLen, 0.5);
	optimVector offsets(nEstLen, 0.5);

	unsigned int const bufferSize = ( 1u << 24 );
	boost::shared_array< char > buffer( new char[ bufferSize  + 1] );

    file.seekg (0, file.beg);
	offsets.push_back( 0 );

//Rprintf( "File read=%d\n", file.gcount() );
    while ( !file.eof() )
    {
       file.read( buffer.get(), bufferSize );
       if ( file.bad() )
            break;

	   offset = 0;
	   while( offset < file.gcount() )
	   {
		   char* szLine = getline( buffer.get(), &offset, file.eof() );
		   if (szLine == NULL)
           {
//Rprintf( "offset/read=%d/%d\n", offset, file.gcount() );
		       if (offset < file.gcount())
		           file.seekg( -1*( file.gcount() - offset ), std::ios_base::cur );

	           break;
           }

//Rprintf( "line=%s\n", szLine );
          int index = -1;
          for (char *p = strtok(szLine," "); p != NULL; p = strtok(NULL, " "))
          {
              char* str = strchr(p, ':');
              if( str == NULL)
			  {
			      if(index==-1)
			         labels.push_back( atoi( szLine ) );
                  continue;
		      }

              *str = '\0';
              index = atoi( p );
              p = str + 1;
              float const value = static_cast< float >( atof( p ) );
              if ( value != 0 ) {
                  values.push_back( value );
                  indices.push_back( index );
                  lines.push_back( labels.size() );

                  if ( index + 1 > static_cast< int >( columns ) )
                     columns = index + 1;
              }
          }
        }

    }

	file.close();
    rows = values.size() + labels.size();
	SEXP mat = PROTECT(allocMatrix(REALSXP, rows, 3));
	double* rmat = REAL(mat);
	for(unsigned int i = 0; i < values.size(); i++)
	{
		rmat[i + rows*0] = (float)lines[i];
		rmat[i + rows*1] = (float)indices[i];
		rmat[i + rows*2] = (float)values[i];
	}

	int nval = values.size();
	for( unsigned int i = 0; i < labels.size(); i++)
	{
		rmat[i + nval + rows*0] = (float)(i+1);
		rmat[i + nval + rows*1] = (float)-1;
		rmat[i + nval + rows*2] = (float)labels[i];
	}

	UNPROTECT(1);

	return(mat);
}
