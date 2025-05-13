#include <rmagine/math/types.h>
#include <rmagine/math/math.h>

namespace rmagine
{

template<typename DataT, unsigned int Rows, unsigned int Cols>
bool contains_nan(Matrix_<DataT, Rows, Cols> M)
{
  for(size_t i=0; i<Rows; i++)
  {
    for(size_t j=0; j<Cols; j++)
    {
      if(M(i,j) != M(i,j)) // only if nan this results in true
      {
        return true;
      }
    }
  }
  
  return false;
}

template<typename DataT, unsigned int Dim>
void chol(
  const Matrix_<DataT, Dim, Dim>& A,
  Matrix_<DataT, Dim, Dim>& L)
{
  for(int i=0; i<Dim; i++) 
  {
    for(int j=i; j<Dim; j++) 
    {
      DataT csum = A(i,j);
      for(int k=i-1; k >= 0; k--)
      {
        csum -= L(i,k) * L(j,k);
      }
      if(i == j)
      {
        if(csum < 0.0)
        {
          // TODO: check if this is OK to do, given the pre-conditions of A
          // if conditions on A are met, this can only happen due to numerical inaccuracies
          csum = 0.0;
        }
        L(i,i) = sqrt(csum);
      }
      else
      {
        const DataT Ltmp = L(i,i);
        if(abs(Ltmp) > 0.00001) // TODO: not hardcode. make dependend of precision
        {
          L(j,i) = csum / Ltmp;
        }
        else
        {
          L(j,i) = 0.0; // this fixed the problem with zero elements on diagonal
        }
      }
    }
  }
  // setZeros for unused part of memory
  for(int i=0; i<Dim; i++)
  {
    for(int j=0; j<i; j++)
    {
      L(j,i) = 0.0;
    }
  }
}


template<typename DataT, unsigned int Rows, unsigned int Cols>
void svd(
    const Matrix_<DataT, Rows, Cols>& a, 
    Matrix_<DataT, Rows, Cols>& u,
    Matrix_<DataT, Cols, Cols>& w,
    Matrix_<DataT, Cols, Cols>& v)
{
    printf("SVD!\n");
    std::cout << "SVD! Template" << std::endl;

    throw std::runtime_error("Template");

    constexpr unsigned int m = Rows;
    constexpr unsigned int n = Cols;

    constexpr unsigned int max_iterations = 30;

    // extra memory required
    bool flag;
    int i, its, j, jj, k, l, nm;
    float anorm, c, f, g, h, s, scale, x, y, z;
    DataT rv1[n];

    g = scale = anorm = 0.0;
    const float eps = std::numeric_limits<float>::epsilon();
    u = a;

    for(i=0; i < n; i++) 
    {
        l = i + 2;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if(i < m) 
        {
            for(k=i; k<m; k++) 
            {
                scale += abs(u(k,i));
            }
            if(scale != 0.0) 
            {
                for(k=i; k<m; k++) 
                {
                    u(k,i) /= scale;
                    s += u(k,i) * u(k,i);
                }
                f = u(i,i);
                g = -sign(sqrt(s),f);
                h = f*g-s;
                u(i,i) = f-g;
                for(j=l-1;j<n;j++) 
                {
                    for (s=0.0,k=i;k<m;k++) 
                    {
                        s += u(k,i) * u(k,j);
                    }
                    f = s/h;
                    for (k=i;k<m;k++)
                    {
                        u(k,j) += f * u(k,i);
                    }
                }
                for (k=i;k<m;k++)
                {
                    u(k,i) *= scale;
                }
            }
        }
        w(i, i) = scale * g;
        g = s = scale = 0.0;
        if(i+1 <= m && i+1 != n)
        {
            for(k=l-1; k<n; k++)
            {
                scale += abs(u(i,k));
            }
            if(scale != 0.0)
            {
                for(k=l-1;k<n;k++) 
                {
                    u(i, k) /= scale;
                    s += u(i,k) * u(i,k);
                }
                f = u(i, l-1);
                g = -sign(sqrt(s),f);
                h = f*g-s;
                u(i,l-1) = f-g;
                for(k=l-1;k<n;k++)
                {
                    rv1[k] = u(i,k) / h;
                }
                for(j=l-1; j<m; j++)
                {
                    for (s=0.0,k=l-1; k<n; k++)
                    {
                        s += u(j,k) * u(i,k);
                    }
                    for (k=l-1; k<n;k++)
                    {
                        u(j,k) += s * rv1[k];
                    }
                }
                for(k=l-1; k<n; k++)
                {
                    u(i,k) *= scale;
                }
            }
        }
        anorm = max(anorm, (abs(w(i, i))+abs(rv1[i])));
    }
    for(i=n-1; i>=0; i--)
    {
        if(i < n-1) 
        {
            if(g != 0.0)
            {
                for(j=l; j<n; j++)
                {
                    v(j,i) = (u(i,j)/u(i,l)) / g;
                }
                for(j=l; j<n; j++)
                {
                    for (s=0.0,k=l;k<n;k++) 
                    {
                        s += u(i,k) * v(k,j);
                    }
                    for (k=l; k<n; k++)
                    {
                        v(k,j) += s * v(k,i);
                    }
                }
            }
            for (j=l; j<n; j++) 
            {
                v(i,j) = 0.0;
                v(j,i) = 0.0;
            }
        }
        v(i,i) = 1.0;
        g = rv1[i];
        l = i;
    }
    for(i=min(m,n)-1; i>=0; i--)
    {
        l = i+1;
        g = w(i, i);
        for(j=l;j<n;j++)
        {
            u(i,j) = 0.0;
        }
        if(g != 0.0)
        {
            g = 1.0/g;
            for(j=l; j<n; j++)
            {
                for (s=0.0,k=l; k<m; k++) 
                {
                    s += u(k,i)*u(k,j);
                }
                f = (s/u(i,i)) * g;
                for (k=i; k<m; k++) 
                {
                    u(k,j) += f * u(k,i);
                }
            }
            for(j=i;j<m;j++) 
            {
                u(j,i) *= g;
            }
        } else {
            for(j=i;j<m;j++) 
            {
                u(j,i) = 0.0;
            } 
        }
        ++u(i,i);
    }
    for(k=n-1; k>=0; k--) 
    {
        for(its=0; its<max_iterations; its++) 
        {
            flag=true;
            for(l=k; l>=0; l--) 
            {
                nm=l-1;
                if (l == 0 || abs(rv1[l]) <= eps*anorm) {
                    flag=false;
                    break;
                }
                if (abs(w(nm, nm)) <= eps*anorm) 
                {
                    break;
                }
            }
            if(flag) 
            {
                c=0.0;
                s=1.0;
                for(i=l; i<k+1; i++) 
                {
                    f = s*rv1[i];
                    rv1[i] = c*rv1[i];
                    if(abs(f) <= eps*anorm) 
                    {
                        break;
                    }
                    g = w(i, i);
                    h = pythag(f,g);
                    w(i, i) = h;
                    h = 1.0/h;
                    c = g*h;
                    s = -f*h;
                    for(j=0; j<m; j++)
                    {
                        y = u(j,nm);
                        z = u(j,i);
                        u(j,nm) = y*c+z*s;
                        u(j,i) = z*c-y*s;
                    }
                }
            }
            z = w(k, k);
            if (l == k)
            {
                if (z < 0.0)
                {
                    w(k,k) = -z;
                    for (j=0;j<n;j++) 
                    {
                        v(j,k) = -v(j,k);
                    }
                }
                break;
            }
            if (its == max_iterations - 1) 
            {
                throw std::runtime_error("no convergence after max svdcmp iterations");
            }
            x = w(l, l);
            nm = k-1;
            y = w(nm, nm);
            g = rv1[nm];
            h = rv1[k];
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g = pythag(f, (DataT)1.0);
            f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
            c = s = 1.0;
            for (j=l; j<=nm; j++) 
            {
                i = j+1;
                g = rv1[i];
                y = w(i, i);
                h = s*g;
                g = c*g;
                z = pythag(f,h);
                rv1[j] = z;
                c = f/z;
                s = h/z;
                f = x*c+g*s;
                g = g*c-x*s;
                h = y*s;
                y *= c;
                for (jj=0;jj<n;jj++)
                {
                    x = v(jj,j);
                    z = v(jj,i);
                    v(jj,j) = x*c+z*s;
                    v(jj,i) = z*c-x*s;
                }
                z = pythag(f,h);
                w(j, j) = z;
                if (z) 
                {
                    z = 1.0/z;
                    c = f*z;
                    s = h*z;
                }
                f = c*g+s*y;
                x = c*y-s*g;
                for (jj=0;jj<m;jj++)
                {
                    y = u(jj,j);
                    z = u(jj,i);
                    u(jj,j) = y*c+z*s;
                    u(jj,i) = z*c-y*s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w(k, k) = x;
        }
    }
}

template<typename DataT, unsigned int Rows, unsigned int Cols>
void svd(
    const Matrix_<DataT, Rows, Cols>& a, 
    Matrix_<DataT, Rows, Cols>& u,
    Matrix_<DataT, Cols, 1>& w, // vector version
    Matrix_<DataT, Cols, Cols>& v)
{
    constexpr unsigned int m = Rows;
    constexpr unsigned int n = Cols;
    constexpr unsigned int max_iterations = 30;
    
    // additional memory required
    bool flag;
    int i, its, j, jj, k, l, nm;
    float anorm, c, f, g, h, s, scale, x, y, z;
    DataT rv1[n];
    
    g = scale = anorm = 0.0;
    float eps = std::numeric_limits<float>::epsilon();
    u = a;

    for(i=0; i < n; i++) 
    {
        l = i+2;
        rv1[i] = scale*g;
        g = s = scale = 0.0;
        if(i < m) 
        {
            for(k=i; k<m; k++) 
            {
                scale += abs(u(k,i));
            }
            if(scale != 0.0) 
            {
                for(k=i; k<m; k++) 
                {
                    u(k,i) /= scale;
                    s += u(k,i) * u(k,i);
                }
                f = u(i,i);
                g = -sign(sqrt(s),f);
                h = f*g-s;
                u(i,i) = f-g;
                for(j=l-1;j<n;j++) 
                {
                    for (s=0.0,k=i;k<m;k++) 
                    {
                        s += u(k,i) * u(k,j);
                    }
                    f = s/h;
                    for (k=i;k<m;k++)
                    {
                        u(k,j) += f * u(k,i);
                    }
                }
                for (k=i;k<m;k++)
                {
                    u(k,i) *= scale;
                }
            }
        }
        w(i, 0) = scale * g;
        g = s = scale = 0.0;
        if(i+1 <= m && i+1 != n)
        {
            for(k=l-1; k<n; k++)
            {
                scale += abs(u(i,k));
            }
            if(scale != 0.0)
            {
                for(k=l-1;k<n;k++) 
                {
                    u(i, k) /= scale;
                    s += u(i,k) * u(i,k);
                }
                f = u(i, l-1);
                g = -sign(sqrt(s),f);
                h = f*g-s;
                
                h = max(h, eps);

                u(i,l-1) = f-g;
                for(k=l-1; k<n; k++)
                {
                    rv1[k] = u(i,k) / h;
                }
                for(j=l-1; j<m; j++)
                {
                    for (s=0.0,k=l-1; k<n; k++)
                    {
                        s += u(j,k) * u(i,k);
                    }
                    for (k=l-1; k<n;k++)
                    {
                        u(j,k) += s * rv1[k];
                    }
                }
                for(k=l-1; k<n; k++)
                {
                    u(i,k) *= scale;
                }
            }
        }
        anorm = max(anorm, (abs(w(i, 0))+abs(rv1[i])));
    }
    for(i=n-1; i>=0; i--)
    {
        if(i < n-1) 
        {
            if(g != 0.0)
            {
                for(j=l; j<n; j++)
                {
                    v(j,i) = (u(i,j)/u(i,l)) / g;
                }
                for(j=l; j<n; j++)
                {
                    for (s=0.0,k=l;k<n;k++) 
                    {
                        s += u(i,k) * v(k,j);
                    }
                    for (k=l; k<n; k++)
                    {
                        v(k,j) += s * v(k,i);
                    }
                }
            }
            for (j=l; j<n; j++) 
            {
                v(i,j) = 0.0;
                v(j,i) = 0.0;
            }
        }
        v(i,i) = 1.0;
        g = rv1[i];
        l = i;
    }
    for(i=min(m,n)-1; i>=0; i--)
    {
        l = i+1;
        g = w(i, 0);
        for(j=l;j<n;j++) 
        {
            u(i,j) = 0.0;
        }
        if(g != 0.0)
        {
            g = 1.0/g;
            for(j=l; j<n; j++)
            {
                for (s=0.0,k=l; k<m; k++) 
                {
                    s += u(k,i)*u(k,j);
                }
                f = (s/u(i,i)) * g;
                for (k=i; k<m; k++) 
                {
                    u(k,j) += f * u(k,i);
                }
            }
            for(j=i;j<m;j++) 
            {
                u(j,i) *= g;
            }
        } else {
            for(j=i;j<m;j++) 
            {
                u(j,i) = 0.0;
            } 
        }
        ++u(i,i);
    }
    for(k=n-1; k>=0; k--) 
    {
        for(its=0; its<max_iterations; its++) 
        {
            flag=true;
            for(l=k; l>=0; l--) 
            {
                nm=l-1;
                if (l == 0 || abs(rv1[l]) <= eps*anorm) {
                    flag=false;
                    break;
                }
                if (abs(w(nm, 0)) <= eps*anorm) 
                {
                    break;
                }
            }
            if(flag) 
            {
                c=0.0;
                s=1.0;
                for(i=l; i<k+1; i++) 
                {
                    f = s*rv1[i];
                    rv1[i] = c*rv1[i];
                    if(abs(f) <= eps*anorm) 
                    {
                        break;
                    }
                    g = w(i, 0);
                    h = pythag(f,g);
                    w(i, 0) = h;
                    h = 1.0/h;
                    c = g*h;
                    s = -f*h;
                    for(j=0; j<m; j++)
                    {
                        y = u(j,nm);
                        z = u(j,i);
                        u(j,nm) = y*c+z*s;
                        u(j,i) = z*c-y*s;
                    }
                }
            }
            z = w(k, 0);
            if (l == k)
            {
                if (z < 0.0)
                {
                    w(k,0) = -z;
                    for (j=0;j<n;j++) 
                    {
                        v(j,k) = -v(j,k);
                    }
                }
                break;
            }
            if (its == max_iterations - 1) 
            {
                throw std::runtime_error("no convergence in 30 svdcmp iterations");
            }
            x = w(l, 0);
            nm = k-1;
            y = w(nm, 0);
            g = rv1[nm];
            h = rv1[k];
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g = pythag(f, (DataT)1.0);
            f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
            c = s = 1.0;
            for (j=l;j<=nm;j++) 
            {
                i = j+1;
                g = rv1[i];
                y = w(i, 0);
                h = s*g;
                g = c*g;
                z = pythag(f,h);
                rv1[j] = z;
                c = f/z;
                s = h/z;
                f = x*c+g*s;
                g = g*c-x*s;
                h = y*s;
                y *= c;
                for (jj=0;jj<n;jj++)
                {
                    x = v(jj,j);
                    z = v(jj,i);
                    v(jj,j) = x*c+z*s;
                    v(jj,i) = z*c-x*s;
                }
                z = pythag(f,h);
                w(j, 0) = z;
                if (z) 
                {
                    z = 1.0/z;
                    c = f*z;
                    s = h*z;
                }
                f = c*g+s*y;
                x = c*y-s*g;
                for (jj=0;jj<m;jj++)
                {
                    y = u(jj,j);
                    z = u(jj,i);
                    u(jj,j) = y*c+z*s;
                    u(jj,i) = z*c-y*s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w(k, 0) = x;
        }
    }
}

} // namespace rmagine
