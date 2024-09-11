#include "rmagine/math/SVD2.hpp"
#include "rmagine/types/Memory.hpp"
#include <assert.h>
// #include <Eigen/Dense>

namespace rmagine
{

SVD2::SVD2()
{
    // eps = std::numeric_limits<float>::epsilon();
    // decompose();
    // reorder();
    // tsh = 0.5 * sqrt(m + n + 1.) * w(0) * eps;
}

SVD2::~SVD2()
{
    
}

void SVD2::decompose(
    const Matrix3x3& a)
{
    bool flag;
    int i, its, j, jj, k, l, nm;
    float anorm,c,f,g,h,s,scale,x,y,z;
    Vector3 rv1;
    g = scale = anorm = 0.0;

    const int m = 3;
    const int n = 3;

    eps = std::numeric_limits<float>::epsilon();
    u = a;

    for(i=0; i < n; i++) 
    {
        l = i+2;
        rv1(i) = scale*g;
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
                g = -SIGN(sqrt(s),f);
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
        w(i) = scale * g;
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
                g = -SIGN(sqrt(s),f);
                h = f*g-s;
                u(i,l-1) = f-g;
                for(k=l-1;k<n;k++)
                {
                    rv1(k) = u(i,k) / h;
                }
                for(j=l-1; j<m; j++)
                {
                    for (s=0.0,k=l-1; k<n; k++)
                    {
                        s += u(j,k) * u(i,k);
                    }
                    for (k=l-1; k<n;k++)
                    {
                        u(j,k) += s * rv1(k);
                    }
                }
                for(k=l-1; k<n; k++)
                {
                    u(i,k) *= scale;
                }
            }
        }
        anorm = MAX(anorm, (abs(w(i))+abs(rv1(i))));
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
        g = rv1(i);
        l = i;
    }
    for(i=MIN(m,n)-1; i>=0; i--)
    {
        l = i+1;
        g = w(i);
        for(j=l;j<n;j++) 
        {
            u(i,j) = 0.0;
        }
        if(g != 0.0) 
        {
            g = 1.0/g;
            for(j=l;j<n;j++)
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
        for(its=0; its<30; its++) 
        {
            flag=true;
            for(l=k; l>=0; l--) 
            {
                nm=l-1;
                if (l == 0 || abs(rv1(l)) <= eps*anorm) {
                    flag=false;
                    break;
                }
                if (abs(w(nm)) <= eps*anorm) 
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
                    f = s*rv1(i);
                    rv1(i) = c*rv1(i);
                    if(abs(f) <= eps*anorm) 
                    {
                        break;
                    }
                    g = w(i);
                    h = PYTHAG(f,g);
                    w(i) = h;
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
            z = w(k);
            if (l == k)
            {
                if (z < 0.0)
                {
                    w(k) = -z;
                    for (j=0;j<n;j++) 
                    {
                        v(j,k) = -v(j,k);
                    }
                }
                break;
            }
            if (its == 29) 
            {
                throw std::runtime_error("no convergence in 30 svdcmp iterations");
            }
            x = w(l);
            nm = k-1;
            y = w(nm);
            g = rv1(nm);
            h = rv1(k);
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
            g = PYTHAG(f, 1.0f);
            f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
            c = s = 1.0;
            for (j=l;j<=nm;j++) 
            {
                i = j+1;
                g = rv1(i);
                y = w(i);
                h = s*g;
                g = c*g;
                z = PYTHAG(f,h);
                rv1(j) = z;
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
                z = PYTHAG(f,h);
                w(j) = z;
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
            rv1(l) = 0.0;
            rv1(k) = f;
            w(k) = x;
        }
    }

    reorder();
    tsh = 0.5 * sqrt(m + n + 1.) * w(0) * eps;
}

void SVD2::reorder()
{
    int i,j,k,s, inc=1;
    float sw;
    Vector3 su, sv;

    const int m = 3;
    const int n = 3;

    do 
    { 
        inc *= 3; 
        inc++; 
    } while (inc <= n);

    do 
    {
        inc /= 3;
        for (i=inc; i<n; i++)
        {
            sw = w(i);
            for (k=0;k<m;k++) 
            {
                su(k) = u(k,i);
            }
            for (k=0;k<n;k++) 
            {
                sv(k) = v(k,i);
            }
            j = i;
            while (w(j-inc) < sw) 
            {
                w(j) = w(j-inc);
                for(k=0; k<m; k++) 
                {
                    u(k,j) = u(k,j-inc);
                }
                for (k=0;k<n;k++)
                {
                    v(k,j) = v(k,j-inc);
                }
                j -= inc;
                if (j < inc)
                {
                    break;
                }
            }
            w(j) = sw;
            for (k=0; k<m; k++)
            {
                u(k,j) = su(k);
            }
            for (k=0; k<n; k++)
            {
                v(k,j) = sv(k);
            }
        }
    } while (inc > 1);

    for(k=0; k<n; k++) 
    {
        s=0;
        for(i=0; i<m; i++) 
        {
            if (u(i,k) < 0.)
            {
                s++;
            }
        }
        for(j=0; j<n; j++) 
        {
            if (v(j,k) < 0.) 
            { 
                s++;
            }
        }
        if(s > (m+n)/2) 
        {
            for (i=0; i<m; i++) 
            {
                u(i,k) = -u(i,k);
            }
            for (j=0;j<n;j++) 
            {
                v(j,k) = -v(j,k);
            }
        }
    }
}

// int SVD2::rank(float thresh)
// {

//     return 0;
// }

// int SVD2::nullity(float thresh)
// {
    
//     return 0;
// }

// Matrix3x3 SVD2::range(float thresh)
// {

//     return Matrix3x3::Zeros();
// }

// Matrix3x3 SVD2::nullspace(float thresh)
// {

//     return Matrix3x3::Zeros();
// }

RMAGINE_FUNCTION
void svd(
    const Matrix3x3& a, 
    Matrix3x3& u,
    Matrix3x3& w,
    Matrix3x3& v)
{
    // TODO: test
    constexpr unsigned int m = 3;
    constexpr unsigned int n = 3;
    constexpr unsigned int max_iterations = 30;
    
    // additional memory required
    bool flag;
    int i, its, j, jj, k, l, nm;
    float anorm, c, f, g, h, s, scale, x, y, z;
    
    Vector3 rv1;
    
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
                g = -SIGN(sqrt(s),f);
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
                g = -SIGN(sqrt(s),f);
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
        anorm = MAX(anorm, (abs(w(i, i))+abs(rv1[i])));
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
    for(i=MIN(m,n)-1; i>=0; i--)
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
                    h = PYTHAG(f,g);
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
                    w(k, k) = -z;
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
            x = w(l, l);
            nm = k-1;
            y = w(nm, nm);
            g = rv1[nm];
            h = rv1[k];
            f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
            g = PYTHAG(f, 1.f);
            f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
            c = s = 1.f;
            for (j=l;j<=nm;j++) 
            {
                i = j+1;
                g = rv1[i];
                y = w(i, i);
                h = s*g;
                g = c*g;
                z = PYTHAG(f,h);
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
                z = PYTHAG(f,h);
                w(j, j) = z;
                if (z) 
                {
                    z = 1.f/z;
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
            rv1[l] = 0.f;
            rv1[k] = f;
            w(k, k) = x;
        }
    }
}


RMAGINE_FUNCTION
void svd2(
    const Matrix3x3& a, 
    Matrix3x3& u,
    Matrix3x3& w,
    Matrix3x3& v)
{
    // TODO: test
    constexpr unsigned int m = 3;
    constexpr unsigned int n = 3;
    constexpr unsigned int max_iterations = 20;
    
    // additional memory required
    bool flag;
    int its, j, jj, k, nm;
    float anorm, c, f, g, h, s, scale, x, y, z;
    
    Vector3 rv1 = Vector3::Zeros();
    
    g = s = scale = anorm = 0.0;
    float eps = std::numeric_limits<float>::epsilon();
    u = a;

    // FIRST PART

    // i = 0;
    // l = 2;
    scale = fabs(u(0,0)) + fabs(u(1,0)) + fabs(u(2,0));
    if(scale > 0.0)
    {
        u(0, 0) /= scale;
        u(1, 0) /= scale;
        u(2, 0) /= scale;

        s = u(0,0) * u(0,0) + u(1,0) * u(1,0) + u(2,0) * u(2,0);
        f = u(0,0);
        g = -SIGN(sqrt(s), f);
        h = f * g - s;

        u(0, 0) = f - g;

        f = (u(0, 0) * u(0, 1) + u(1, 0) * u(1, 1) + u(2, 0) * u(2, 1)) / h;
        u(0, 1) += f * u(0, 0);
        u(1, 1) += f * u(1, 0);
        u(2, 1) += f * u(2, 0);

        f = (u(0, 0) * u(0, 2) + u(1, 0) * u(1, 2) + u(2, 0) * u(2, 2)) / h;
        u(0, 2) += f * u(0, 0);
        u(1, 2) += f * u(1, 0);
        u(2, 2) += f * u(2, 0);

        u(0, 0) *= scale;
        u(1, 0) *= scale;
        u(2, 0) *= scale;
    }
    
    w(0, 0) = scale * g;
    g = s = scale = 0.0;
    
    scale = abs(u(0,0)) + abs(u(0,1)) + abs(u(0,2));
    
    if(scale > 0.0)
    {
        u(0, 1) /= scale;
        u(0, 2) /= scale;
        s = u(0,1) * u(0,1) + u(0,2) * u(0,2);

        f = u(0, 1);
        g = -SIGN(sqrt(s),f);
        h = f * g-s;
        u(0, 1) = f - g;

        rv1.y = u(0, 1) / h;
        rv1.z = u(0, 2) / h;

        s = u(1,1) * u(0,1) + u(1,2) * u(0,2);
        u(1, 1) += s * rv1.y;
        u(1, 2) += s * rv1.z;
    
        s = u(2,1) * u(0,1) + u(2,2) * u(0,2);
        u(2, 1) += s * rv1.y;
        u(2, 2) += s * rv1.z;            

        u(0, 1) *= scale;
        u(0, 2) *= scale;
    }
    
    anorm = fabs(w(0, 0));
    // anorm = MAX(anorm, (fabs(w(0, 0)) + fabs(rv1.x))); // rv1.x is always 0 here, anorm too. fabs(X) >= 0
    
    // i = 1;
    // l = 3;
    rv1.y = scale * g;
    g = 0.0;
    scale = fabs(u(1, 1)) + fabs(u(2, 1));
    
    if(scale > 0.0)
    {
        u(1,1) /= scale;
        u(2,1) /= scale;

        s = u(1,1) * u(1,1) + u(2,1) * u(2,1);
        f = u(1,1);
        g = -SIGN(sqrt(s),f);
        h = f * g - s;
        u(1,1) = f-g;
        
        f = (u(1,1) * u(1,2) + u(2,1) * u(2,2)) / h;
        u(1,2) += f * u(1,1);
        u(2,2) += f * u(2,1);
        
        u(1,1) *= scale;
        u(2,1) *= scale;
    }
    
    w(1, 1) = scale * g;
    g = s = scale = 0.0;
    
    scale = abs(u(1,2));
    if(scale > 0.0)
    {
        u(1,2) /= scale;
        s = u(1,2) * u(1,2);
        
        f = u(1, 2);
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        u(1,2) = f - g;

        rv1.z = u(1,2) / h;
        s = u(2,2) * u(1,2);

        u(2,2) += s * rv1.z;
        u(1,2) *= scale;
    }

    anorm = MAX(anorm, (abs(w(1, 1)) + abs(rv1.y)));
    
    rv1.z = scale * g;

    scale = abs(u(2, 2));
    if(scale > 0.0) 
    {
        u(2, 2) /= scale;
        s = u(2, 2) * u(2, 2);
        f = u(2, 2);
        g = -SIGN(sqrt(s),f);
        h = f * g - s;

        u(2, 2) = f - g;
        u(2, 2) *= scale;
    }
    
    w(2, 2) = scale * g;
    g = s = scale = 0.0;
    
    anorm = MAX(anorm, (abs(w(2, 2))+abs(rv1.z)));

    // SECOND PART    
    v(2, 2) = 1.0;
    g = rv1.z;

    // i = 1;
    // l = 2;
    if(fabs(g) > 0.0)
    {
        v(2,1) = (u(1,2) / u(1,2)) / g;
        s = u(1,2) * v(2,2);
        v(2,2) += s * v(2,1);
    }
    v(1,2) = 0.0;
    v(2,1) = 0.0;
    v(1,1) = 1.0;

    g = rv1.y;

    // l = 1;
    // i = 0;
    if(fabs(g) > 0.0)
    {
        v(1,0) = (u(0,1) / u(0,1)) / g;
        v(2,0) = (u(0,2) / u(0,1)) / g;

        s = u(0,1) * v(1,1) + u(0,2) * v(2,1);
        v(1,1) += s * v(1,0);
        v(2,1) += s * v(2,0);

        s = u(0,1) * v(1,2) + u(0,2) * v(2,2);
        v(1,2) += s * v(1,0);
        v(2,2) += s * v(2,0);
    }
    v(0,1) = 0.0;
    v(1,0) = 0.0;
    v(0,2) = 0.0;
    v(2,0) = 0.0;
    v(0,0) = 1.0;
    g = rv1.x;


    // THIRD PART
    
    // i = 2;
    // l = 3;
    g = w(2, 2);
    if(fabs(g) > 0.0)
    {
        u(2,2) /= g;
    } else { 
        // TODO(amock): shouldnt this be a large number?
        u(2,2) = 0.0;   
    }
    u(2,2) += 1.0;
    
    // i = 1;
    // l = 2;

    g = w(1, 1);
    u(1,2) = 0.0;
    
    if(fabs(g) > 0.0)
    {
        g = 1.0/g;
        s = u(2,1) * u(2,2);
        f = (s/u(1,1)) * g;

        u(1,2) += f * u(1,1);
        u(2,2) += f * u(2,1);
    
        u(1,1) *= g;
        u(2,1) *= g;
    } else {
        u(1,1) = 0.0;
        u(2,1) = 0.0;
    }
    u(1,1) += 1.0;
    
    // i = 0;
    // l = 1;
    g = w(0, 0);
    u(0,1) = 0.0;
    u(0,2) = 0.0;
    
    if(fabs(g) > 0.0)
    {
        f = (u(1,0) * u(1,1) + u(2,0) * u(2,1)) / (g * u(0,0));
        u(0,1) += f * u(0,0);
        u(1,1) += f * u(1,0);
        u(2,1) += f * u(2,0);

        f = (u(1,0) * u(1,2) + u(2,0) * u(2,2)) / (g * u(0,0));
        u(0,2) += f * u(0,0);
        u(1,2) += f * u(1,0);
        u(2,2) += f * u(2,0);

        u(0,0) /= g;
        u(1,0) /= g;
        u(2,0) /= g;
    } else {
        u(0,0) = 0.0;
        u(1,0) = 0.0;
        u(2,0) = 0.0;
    }
    u(0,0) += 1.0;

    int i, l;

    // PART 4: Opti

    // k = 2;
    for(its=0; its<max_iterations; its++) 
    {
        // flag=true;
        // l = 2;
        // if(MIN(fabs(rv1.z), fabs(w(1,1))) > eps*anorm)
        // {
        //     l = 1;
        //     if(MIN(fabs(rv1.y),abs(w(0,0))) > eps*anorm)
        //     {
        //         l = 0;
        //     }
        // }

        flag=true;
        l=2;
        if(abs(rv1.z) <= eps*anorm)
        {
            flag=false;
        }
        else if(abs(w(1,1)) > eps*anorm)
        {
            l=1;
            if(abs(rv1.y) <= eps*anorm) 
            {
                flag=false;
            }
            else if(abs(w(0,0)) > eps*anorm) 
            {
                l=0;
                flag = false;
            }
        }
        
        if(flag)
        {
            c=0.0;
            s=1.0;
            for(i=l; i<3; i++) 
            {
                f = s*rv1[i];
                rv1[i] = c*rv1[i];
                if(abs(f) <= eps*anorm) 
                {
                    break;
                }
                g = w(i, i);
                h = PYTHAG(f,g);
                w(i, i) = h;
                h = 1.0/h;
                c = g*h;
                s = -f*h;
                for(j=0; j<m; j++)
                {
                    y = u(j,l-1);
                    z = u(j,i);
                    u(j,l-1) = y*c+z*s;
                    u(j,i) = z*c-y*s;
                }
            }
        }
        z = w(2,2);
        if(l == 2)
        {
            if(z < 0.0)
            {
                w(2, 2) = -z;
                for (j=0; j<3; j++) 
                {
                    v(j,2) = -v(j,2);
                }
            }
            break;
        }
        if(its == max_iterations - 1) 
        {
            // std::cout << "no convergence in " << max_iterations << " svdcmp iterations" << std::endl;
            throw std::runtime_error("no convergence in max svdcmp iterations");
        }
        x = w(l,l);
        y = w(1,1);
        g = rv1.y;
        h = rv1.z;
        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
        g = PYTHAG(f, 1.f);
        f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
        c = s = 1.f;
        for (j=l; j<2; j++) 
        {
            i = j+1;
            g = rv1[i];
            y = w(i, i);
            h = s*g;
            g = c*g;
            z = PYTHAG(f,h);
            rv1[j] = z;
            c = f/z;
            s = h/z;
            f = x*c+g*s;
            g = g*c-x*s;
            h = y*s;
            y *= c;
            for(jj=0;jj<3;jj++)
            {
                x = v(jj,j);
                z = v(jj,i);
                v(jj,j) = x*c+z*s;
                v(jj,i) = z*c-x*s;
            }
            z = PYTHAG(f,h);
            w(j,j) = z;
            if(z>0.0) 
            {
                z = 1.f/z;
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
        rv1[l] = 0.f;
        rv1.z = f;
        w(2,2) = x;
    }


    for(its=0; its<max_iterations; its++) 
    {
        flag=true;
        l=1;
        if(abs(rv1.y) <= eps*anorm) 
        {
            flag=false;
        }
        else if(abs(w(0,0)) > eps*anorm) 
        {
            l=0;
            flag = false;
        }

        if(flag)
        {
            c=0.0;
            s=1.0;
            for(i=l; i<2; i++) 
            {
                f = s*rv1[i];
                rv1[i] = c*rv1[i];
                if(abs(f) <= eps*anorm) 
                {
                    break;
                }
                g = w(i,i);
                h = PYTHAG(f,g);
                w(i, i) = h;
                h = 1.0/h;
                c = g*h;
                s = -f*h;
                for(j=0; j<m; j++)
                {
                    y = u(j,l-1);
                    z = u(j,i);
                    u(j,l-1) = y*c+z*s;
                    u(j,i) = z*c-y*s;
                }
            }
        }
        z = w(1,1);
        if(l == 1)
        {
            if(z < 0.0)
            {
                w(1,1) = -z;
                for (j=0; j<3; j++) 
                {
                    v(j,1) = -v(j,1);
                }
            }
            break;
        }
        if(its == max_iterations - 1) 
        {
            std::cout << "no convergence in " << max_iterations << " svdcmp iterations" << std::endl;
            throw std::runtime_error("no convergence in max svdcmp iterations");
        }

        x = w(l, l);
        y = w(0, 0);
        g = rv1.x;
        h = rv1.y;
        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
        g = PYTHAG(f, 1.f);
        f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
        c = s = 1.f;

        if(l == 0)
        {
            g = rv1.y;
            y = w(1,1);
            h = s*g;
            g = c*g;
            z = PYTHAG(f,h);
            rv1.x = z;
            c = f/z;
            s = h/z;
            f = x*c+g*s;
            g = g*c-x*s;
            h = y*s;
            y *= c;


            x = v(0,0);
            z = v(0,1);
            v(0,0) = x*c+z*s;
            v(0,1) = z*c-x*s;

            x = v(1,0);
            z = v(1,1);
            v(1,0) = x*c+z*s;
            v(1,1) = z*c-x*s;

            x = v(2,0);
            z = v(2,1);
            v(2,0) = x*c+z*s;
            v(2,1) = z*c-x*s;


            z = PYTHAG(f,h);
            w(0,0) = z;
            if(z>0.f)
            {
                z = 1.f/z;
                c = f*z;
                s = h*z;
            }
            f = c*g+s*y;
            x = c*y-s*g;
            for(jj=0; jj<3; jj++)
            {
                y = u(jj,0);
                z = u(jj,1);
                u(jj,0) = y*c+z*s;
                u(jj,1) = z*c-y*s;
            }
        }
        
        rv1[l] = 0.f;
        rv1.y = f;
        w(1,1) = x;
    }


    z = w(0,0);
    if (z < 0.0)
    {
        w(0,0) = -z;
        v(0,0) = -v(0,0);
        v(1,0) = -v(1,0);
        v(2,0) = -v(2,0);
    }
}



// RMAGINE_FUNCTION
// void svd(
//     const Matrix3x3& a, 
//     Matrix3x3& u,
//     Vector3& w, // matrix
//     Matrix3x3& v)
// {
//     // TODO: test
//     constexpr unsigned int m = 3;
//     constexpr unsigned int n = 3;
//     constexpr unsigned int max_iterations = 30;
    
//     // additional memory required
//     bool flag;
//     int i, its, j, jj, k, l, nm;
//     float anorm, c, f, g, h, s, scale, x, y, z;
    
//     Vector3 rv1;
    
//     g = scale = anorm = 0.0;
//     float eps = std::numeric_limits<float>::epsilon();
//     u = a;

//     for(i=0; i < n; i++) 
//     {
//         l = i+2;
//         rv1[i] = scale*g;
//         g = s = scale = 0.0;
//         if(i < m) 
//         {
//             for(k=i; k<m; k++) 
//             {
//                 scale += abs(u(k,i));
//             }
//             if(scale != 0.0) 
//             {
//                 for(k=i; k<m; k++) 
//                 {
//                     u(k,i) /= scale;
//                     s += u(k,i) * u(k,i);
//                 }
//                 f = u(i,i);
//                 g = -SIGN(sqrt(s),f);
//                 h = f*g-s;
//                 u(i,i) = f-g;
//                 for(j=l-1;j<n;j++) 
//                 {
//                     for (s=0.0,k=i;k<m;k++) 
//                     {
//                         s += u(k,i) * u(k,j);
//                     }
//                     f = s/h;
//                     for (k=i;k<m;k++)
//                     {
//                         u(k,j) += f * u(k,i);
//                     }
//                 }
//                 for (k=i;k<m;k++)
//                 {
//                     u(k,i) *= scale;
//                 }
//             }
//         }
//         w(i) = scale * g;
//         g = s = scale = 0.0;
//         if(i+1 <= m && i+1 != n)
//         {
//             for(k=l-1; k<n; k++)
//             {
//                 scale += abs(u(i,k));
//             }
//             if(scale != 0.0)
//             {
//                 for(k=l-1;k<n;k++) 
//                 {
//                     u(i, k) /= scale;
//                     s += u(i,k) * u(i,k);
//                 }
//                 f = u(i, l-1);
//                 g = -SIGN(sqrt(s),f);
//                 h = f*g-s;
//                 u(i,l-1) = f-g;
//                 for(k=l-1;k<n;k++)
//                 {
//                     rv1[k] = u(i,k) / h;
//                 }
//                 for(j=l-1; j<m; j++)
//                 {
//                     for (s=0.0,k=l-1; k<n; k++)
//                     {
//                         s += u(j,k) * u(i,k);
//                     }
//                     for (k=l-1; k<n;k++)
//                     {
//                         u(j,k) += s * rv1[k];
//                     }
//                 }
//                 for(k=l-1; k<n; k++)
//                 {
//                     u(i,k) *= scale;
//                 }
//             }
//         }
//         anorm = MAX(anorm, (abs(w(i))+abs(rv1[i])));
//     }
//     for(i=n-1; i>=0; i--)
//     {
//         if(i < n-1) 
//         {
//             if(g != 0.0)
//             {
//                 for(j=l; j<n; j++)
//                 {
//                     v(j,i) = (u(i,j)/u(i,l)) / g;
//                 }
//                 for(j=l; j<n; j++)
//                 {
//                     for (s=0.0,k=l;k<n;k++) 
//                     {
//                         s += u(i,k) * v(k,j);
//                     }
//                     for (k=l; k<n; k++)
//                     {
//                         v(k,j) += s * v(k,i);
//                     }
//                 }
//             }
//             for (j=l; j<n; j++) 
//             {
//                 v(i,j) = 0.0;
//                 v(j,i) = 0.0;
//             }
//         }
//         v(i,i) = 1.0;
//         g = rv1[i];
//         l = i;
//     }
//     for(i=MIN(m,n)-1; i>=0; i--)
//     {
//         l = i+1;
//         g = w(i);
//         for(j=l;j<n;j++) 
//         {
//             u(i,j) = 0.0;
//         }
//         if(g != 0.0)
//         {
//             g = 1.0/g;
//             for(j=l; j<n; j++)
//             {
//                 for (s=0.0,k=l; k<m; k++) 
//                 {
//                     s += u(k,i)*u(k,j);
//                 }
//                 f = (s/u(i,i)) * g;
//                 for (k=i; k<m; k++) 
//                 {
//                     u(k,j) += f * u(k,i);
//                 }
//             }
//             for(j=i;j<m;j++) 
//             {
//                 u(j,i) *= g;
//             }
//         } else {
//             for(j=i;j<m;j++) 
//             {
//                 u(j,i) = 0.0;
//             } 
//         }
//         ++u(i,i);
//     }
//     for(k=n-1; k>=0; k--) 
//     {
//         for(its=0; its<max_iterations; its++) 
//         {
//             flag=true;
//             for(l=k; l>=0; l--) 
//             {
//                 nm=l-1;
//                 if (l == 0 || abs(rv1[l]) <= eps*anorm) {
//                     flag=false;
//                     break;
//                 }
//                 if (abs(w(nm)) <= eps*anorm) 
//                 {
//                     break;
//                 }
//             }
//             if(flag) 
//             {
//                 c=0.0;
//                 s=1.0;
//                 for(i=l; i<k+1; i++) 
//                 {
//                     f = s*rv1[i];
//                     rv1[i] = c*rv1[i];
//                     if(abs(f) <= eps*anorm) 
//                     {
//                         break;
//                     }
//                     g = w(i);
//                     h = PYTHAG(f,g);
//                     w(i) = h;
//                     h = 1.0/h;
//                     c = g*h;
//                     s = -f*h;
//                     for(j=0; j<m; j++)
//                     {
//                         y = u(j,nm);
//                         z = u(j,i);
//                         u(j,nm) = y*c+z*s;
//                         u(j,i) = z*c-y*s;
//                     }
//                 }
//             }
//             z = w(k);
//             if (l == k)
//             {
//                 if (z < 0.0)
//                 {
//                     w(k) = -z;
//                     for (j=0;j<n;j++) 
//                     {
//                         v(j,k) = -v(j,k);
//                     }
//                 }
//                 break;
//             }
//             if (its == max_iterations - 1) 
//             {
//                 throw std::runtime_error("no convergence in 30 svdcmp iterations");
//             }
//             x = w(l);
//             nm = k-1;
//             y = w(nm);
//             g = rv1[nm];
//             h = rv1[k];
//             f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
//             g = PYTHAG(f, 1.f);
//             f = ((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
//             c = s = 1.f;
//             for (j=l;j<=nm;j++) 
//             {
//                 i = j+1;
//                 g = rv1[i];
//                 y = w(i);
//                 h = s*g;
//                 g = c*g;
//                 z = PYTHAG(f,h);
//                 rv1[j] = z;
//                 c = f/z;
//                 s = h/z;
//                 f = x*c+g*s;
//                 g = g*c-x*s;
//                 h = y*s;
//                 y *= c;
//                 for (jj=0;jj<n;jj++)
//                 {
//                     x = v(jj,j);
//                     z = v(jj,i);
//                     v(jj,j) = x*c+z*s;
//                     v(jj,i) = z*c-x*s;
//                 }
//                 z = PYTHAG(f,h);
//                 w(j) = z;
//                 if (z) 
//                 {
//                     z = 1.f/z;
//                     c = f*z;
//                     s = h*z;
//                 }
//                 f = c*g+s*y;
//                 x = c*y-s*g;
//                 for (jj=0;jj<m;jj++)
//                 {
//                     y = u(jj,j);
//                     z = u(jj,i);
//                     u(jj,j) = y*c+z*s;
//                     u(jj,i) = z*c-y*s;
//                 }
//             }
//             rv1[l] = 0.f;
//             rv1[k] = f;
//             w(k) = x;
//         }
//     }
// }

} // namespace rmagine
