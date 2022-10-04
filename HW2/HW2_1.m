clear all
close all

%% Pricing of a compound option 

V_t=100;
sigma_t=.3;
D=70;
r=.04;
T=10;
t=0;
d_1=(log(V_t/D)+(r+sigma_t^2/2)*(T-t))/(sqrt(T-t)*sigma_t);
d_2=d_1-sqrt(T-t)*sigma_t;
PV=V_t*normcdf(d_1)-exp(-r*(T-t))*D*normcdf(d_2);
b_1=d_1;
b_2=d_2;
strikes=PV*[.6 .8 1 1.2];
tau=[2 4 6 8];
prices=zeros(length(tau),length(strikes));
startguess_S=V_t;
for i=1:length(tau)
    
    rho=sqrt((tau(i)-t)/(T-t));
    cov=[1,rho;rho,1];
    for j=1:length(strikes)
 
        S_=newton_rhapson_S(strikes(j),tau(i),D,r,sigma_t,T,startguess_S);
        a_1=(log(V_t/S_)+(r+sigma_t^2/2)*(tau(i)-t))/(sqrt(tau(i)-t)*sigma_t);
        a_2=a_1-sqrt(tau(i)-t)*sigma_t;
        prices(i,j)=V_t*mvncdf([a_1,b_1],[0,0],cov)-D*exp(-r*(T-t))*mvncdf([a_2,b_2],[0,0],cov)-exp(-r*(tau(i)-t))*strikes(j)*normcdf(a_2);
    end
end
sympref('FloatingPointOutput',1);
latex(sym(prices));

%% Implied volatility of the firm's equity

implied_volatility=zeros(length(tau),length(strikes));
startguess_V=sigma_t;
for i=1:length(tau)
    for j=1:length(strikes)
        implied_volatility(i,j)=newton_rhapson_vol(PV,strikes(j),prices(i,j),tau(i),r,startguess_V);
    end
end
latex(sym(implied_volatility))

%% Plot  IV vs strike for each maturity 

for i=1:length(tau)
    subplot(2,2,i)
    plot(strikes,implied_volatility(i,:))
    xlabel('Strike')
    ylabel('Implied volatility')
    title('$\tau$='+string(tau(i)),'Interpreter','latex')
    grid on
end


%% Newton-Rhapson method on stock and volatility

function solution=newton_rhapson_S(K,tau,D,r,sigma_t,T,startguess)
x = startguess;
x_old=.1;
iter = 0;
while abs(x_old-x) > 1e-8  
    x_old = x;  
    x = x - (x*normcdf((log(x/D)+(r+sigma_t^2/2)*(T-tau))/(sqrt(T-tau)*sigma_t))-D*exp(-r*(T-tau))*normcdf((log(x/D)+(r-sigma_t^2/2)*(T-tau))/(sqrt(T-tau)*sigma_t))-K)/(normcdf((log(x/D)+(r+sigma_t^2/2)*(T-tau))/(sqrt(T-tau)*sigma_t)));  
end 
solution=x;
end


function solution=newton_rhapson_vol(S,K_1,K_2,tau,r,startguess)
x = startguess;
x_old=100;
iter = 0;
while abs(x_old-x) > 1e-8  
    x_old = x;  
    x = x - (S*normcdf((log(S/K_1)+(r+x^2/2)*(tau))/(sqrt(tau)*x))-K_1*exp(-r*(tau))*normcdf((log(S/K_1)+ ...
        (r-x^2/2)*(tau))/(sqrt(tau)*x))-K_2)/(sqrt(tau)*S*normpdf((log(S/K_1)+(r+x^2/2)*(tau))/(sqrt(tau)*x)));  
end 
solution=x;
end

