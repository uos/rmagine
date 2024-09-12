#include <rmagine/math/types.h>
#include <rmagine/math/linalg.h>
#include <cuda_runtime.h>

namespace rmagine
{

RMAGINE_DEVICE_FUNCTION
void svd(
    const Matrix3x3& a, 
    Matrix3x3& u,
    Matrix3x3& w,
    Matrix3x3& v)
{
    // printf("SVDD\n");

    
    // TODO: test
    const unsigned int max_iterations = 20;
    
    // additional memory required
    bool flag;
    int its, j, jj;
    float anorm, c, f, g, h, s, scale, x, y, z;
    
    Vector3 rv1;
    rv1.x = 0.0;
    rv1.y = 0.0;
    rv1.z = 0.0;

    
    
    g = s = scale = anorm = 0.0;
    const float eps = __FLT_EPSILON__;
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
        g = -sign(sqrt(s), f);
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
        g = -sign(sqrt(s),f);
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
    // anorm = max(anorm, (fabs(w(0, 0)) + fabs(rv1.x))); // rv1.x is always 0 here, anorm too. fabs(X) >= 0
    
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
        g = -sign(sqrt(s),f);
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
        g = -sign(sqrt(s), f);
        h = f * g - s;
        u(1,2) = f - g;

        rv1.z = u(1,2) / h;
        s = u(2,2) * u(1,2);

        u(2,2) += s * rv1.z;
        u(1,2) *= scale;
    }

    anorm = max(anorm, (abs(w(1, 1)) + abs(rv1.y)));
    
    

    rv1.z = scale * g;

    scale = abs(u(2, 2));
    if(scale > 0.0) 
    {
        u(2, 2) /= scale;
        s = u(2, 2) * u(2, 2);
        f = u(2, 2);
        g = -sign(sqrt(s),f);
        h = f * g - s;

        u(2, 2) = f - g;
        u(2, 2) *= scale;
    }
    
    w(2, 2) = scale * g;
    g = s = scale = 0.0;
    
    anorm = max(anorm, (abs(w(2, 2))+abs(rv1.z)));

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
    for(int its=0; its<max_iterations; its++) 
    {
        // printf("First Iter %d/%d\n", its, max_iterations);
        // return;

        flag=true;
        l=2;
        if(fabs(rv1.z) <= eps*anorm)
        {
            flag=false;
        }
        else if(fabs(w(1,1)) > eps*anorm)
        {
            l=1;
            if(fabs(rv1.y) <= eps*anorm) 
            {
                flag=false;
            }
            else if(fabs(w(0,0)) > eps*anorm) 
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
                if(fabs(f) <= eps*anorm) 
                {
                    break;
                }
                g = w(i, i);
                h = pythag(f,g);
                w(i, i) = h;
                h = 1.0/h;
                c = g*h;
                s = -f*h;
                for(j=0; j<3; j++)
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
            // throw std::runtime_error("no convergence in max svdcmp iterations");
        }
        x = w(l,l);
        y = w(1,1);
        g = rv1.y;
        h = rv1.z;
        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
        g = pythag(f, 1.f);
        f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
        c = s = 1.f;
        for (j=l; j<2; j++) 
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
            for(jj=0;jj<3;jj++)
            {
                x = v(jj,j);
                z = v(jj,i);
                v(jj,j) = x*c+z*s;
                v(jj,i) = z*c-x*s;
            }
            z = pythag(f,h);
            w(j,j) = z;
            if(z>0.0) 
            {
                z = 1.f/z;
                c = f*z;
                s = h*z;
            }
            f = c*g+s*y;
            x = c*y-s*g;
            for (jj=0;jj<3;jj++)
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
                h = pythag(f,g);
                w(i, i) = h;
                h = 1.0/h;
                c = g*h;
                s = -f*h;
                for(j=0; j<3; j++)
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
            // std::cout << "no convergence in " << max_iterations << " svdcmp iterations" << std::endl;
            // throw std::runtime_error("no convergence in max svdcmp iterations");
        }

        x = w(l, l);
        y = w(0, 0);
        g = rv1.x;
        h = rv1.y;
        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
        g = pythag(f, 1.f);
        f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
        c = s = 1.f;

        if(l == 0)
        {
            g = rv1.y;
            y = w(1,1);
            h = s*g;
            g = c*g;
            z = pythag(f,h);
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


            z = pythag(f,h);
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


RMAGINE_DEVICE_FUNCTION
void svd(
    const Matrix3x3& a, 
    Matrix3x3& u,
    Vector3& w,
    Matrix3x3& v)
{
    // TODO: test
    // constexpr unsigned int m = 3;
    // constexpr unsigned int n = 3;
    constexpr unsigned int max_iterations = 20;
    
    // additional memory required
    bool flag;
    int its, j, jj;
    float anorm, c, f, g, h, s, scale, x, y, z;
    
    Vector3 rv1 = Vector3::Zeros();
    
    g = s = scale = anorm = 0.0;
    float eps = __FLT_EPSILON__;
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
        g = -sign(sqrt(s), f);
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
    
    w.x = scale * g;
    g = s = scale = 0.0;
    
    scale = abs(u(0,0)) + abs(u(0,1)) + abs(u(0,2));
    
    if(scale > 0.0)
    {
        u(0, 1) /= scale;
        u(0, 2) /= scale;
        s = u(0,1) * u(0,1) + u(0,2) * u(0,2);

        f = u(0, 1);
        g = -sign(sqrt(s),f);
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
    
    anorm = fabs(w.x);
    // anorm = max(anorm, (fabs(w(0, 0)) + fabs(rv1.x))); // rv1.x is always 0 here, anorm too. fabs(X) >= 0
    
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
        g = -sign(sqrt(s),f);
        h = f * g - s;
        u(1,1) = f-g;
        
        f = (u(1,1) * u(1,2) + u(2,1) * u(2,2)) / h;
        u(1,2) += f * u(1,1);
        u(2,2) += f * u(2,1);
        
        u(1,1) *= scale;
        u(2,1) *= scale;
    }
    
    w.y = scale * g;
    g = s = scale = 0.0;
    
    scale = abs(u(1,2));
    if(scale > 0.0)
    {
        u(1,2) /= scale;
        s = u(1,2) * u(1,2);
        
        f = u(1, 2);
        g = -sign(sqrt(s), f);
        h = f * g - s;
        u(1,2) = f - g;

        rv1.z = u(1,2) / h;
        s = u(2,2) * u(1,2);

        u(2,2) += s * rv1.z;
        u(1,2) *= scale;
    }

    anorm = max(anorm, (abs(w.y) + abs(rv1.y)));
    
    rv1.z = scale * g;

    scale = abs(u(2, 2));
    if(scale > 0.0) 
    {
        u(2, 2) /= scale;
        s = u(2, 2) * u(2, 2);
        f = u(2, 2);
        g = -sign(sqrt(s),f);
        h = f * g - s;

        u(2, 2) = f - g;
        u(2, 2) *= scale;
    }
    
    w.z = scale * g;
    g = s = scale = 0.0;
    
    anorm = max(anorm, (abs(w.z)+abs(rv1.z)));

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
    g = w.z;
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

    g = w.y;
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
    g = w.x;
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
        // if(min(fabs(rv1.z), fabs(w(1,1))) > eps*anorm)
        // {
        //     l = 1;
        //     if(min(fabs(rv1.y),abs(w(0,0))) > eps*anorm)
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
        else if(abs(w.y) > eps*anorm)
        {
            l=1;
            if(abs(rv1.y) <= eps*anorm) 
            {
                flag=false;
            }
            else if(abs(w.x) > eps*anorm) 
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
                g = w(i);
                h = pythag(f,g);
                w(i) = h;
                h = 1.0/h;
                c = g*h;
                s = -f*h;
                for(j=0; j<3; j++)
                {
                    y = u(j,l-1);
                    z = u(j,i);
                    u(j,l-1) = y*c+z*s;
                    u(j,i) = z*c-y*s;
                }
            }
        }
        z = w.z;
        if(l == 2)
        {
            if(z < 0.0)
            {
                w.z = -z;
                v(0,2) = -v(0,2);
                v(1,2) = -v(1,2);
                v(2,2) = -v(2,2);
            }
            break;
        }
        if(its == max_iterations - 1) 
        {
            // std::cout << "no convergence in " << max_iterations << " svdcmp iterations" << std::endl;
            // throw std::runtime_error("no convergence in max svdcmp iterations");
        }
        x = w(l);
        y = w.y;
        g = rv1.y;
        h = rv1.z;
        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
        g = pythag(f, 1.f);
        f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
        c = s = 1.f;
        for (j=l; j<2; j++)
        {
            i = j+1;
            g = rv1[i];
            y = w(i);
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
            for(jj=0;jj<3;jj++)
            {
                x = v(jj,j);
                z = v(jj,i);
                v(jj,j) = x*c+z*s;
                v(jj,i) = z*c-x*s;
            }
            z = pythag(f,h);
            w(j) = z;
            if(z>0.0) 
            {
                z = 1.f/z;
                c = f*z;
                s = h*z;
            }
            f = c*g+s*y;
            x = c*y-s*g;
            for (jj=0;jj<3;jj++)
            {
                y = u(jj,j);
                z = u(jj,i);
                u(jj,j) = y*c+z*s;
                u(jj,i) = z*c-y*s;
            }
        }
        rv1[l] = 0.f;
        rv1.z = f;
        w.z = x;
    }


    for(its=0; its<max_iterations; its++) 
    {
        flag=true;
        l=1;
        if(abs(rv1.y) <= eps*anorm) 
        {
            flag=false;
        }
        else if(abs(w.x) > eps*anorm) 
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
                g = w(i);
                h = pythag(f,g);
                w(i) = h;
                h = 1.0/h;
                c = g*h;
                s = -f*h;
                for(j=0; j<3; j++)
                {
                    y = u(j,l-1);
                    z = u(j,i);
                    u(j,l-1) = y*c+z*s;
                    u(j,i) = z*c-y*s;
                }
            }
        }
        z = w.y;
        if(l == 1)
        {
            if(z < 0.0)
            {
                w.y = -z;
                for (j=0; j<3; j++) 
                {
                    v(j,1) = -v(j,1);
                }
            }
            break;
        }
        if(its == max_iterations - 1) 
        {
        //     std::cout << "no convergence in " << max_iterations << " svdcmp iterations" << std::endl;
        //     throw std::runtime_error("no convergence in max svdcmp iterations");
        }

        x = w(l);
        y = w.x;
        g = rv1.x;
        h = rv1.y;
        f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
        g = pythag(f, 1.f);
        f = ((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x;
        c = s = 1.f;

        if(l == 0)
        {
            g = rv1.y;
            y = w.y;
            h = s*g;
            g = c*g;
            z = pythag(f,h);
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


            z = pythag(f,h);
            w.x = z;
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
        w.y = x;
    }


    z = w.x;
    if (z < 0.0)
    {
        w.x = -z;
        v(0,0) = -v(0,0);
        v(1,0) = -v(1,0);
        v(2,0) = -v(2,0);
    }
}

} // namespace rmagine
