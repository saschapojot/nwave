from sympy import *

s,p,d=symbols("s,p,d",cls=Symbol,positive=True)

x,u=symbols("x,u",cls=Symbol,positive=True)

f=symbols("f",cls=Function)

phi=x**((d-1)/2)*f(u,x)

dt2phi=diff(phi,(u,2))

drphi=-x**2*diff(phi,x)-diff(phi,u)

dr2phi=4*x**3*diff(phi,x)+x**4*diff(phi,(x,2))\
    +2*x**2*diff(diff(phi,x),u)+diff(phi,(u,2))

eqn=dt2phi-dr2phi-(d-1)*x*drphi-phi**(2*p+1)

pprint(expand(eqn/x**(d/2+Rational(3,2))))