%%%  RUN STOCHASTICALLY  

%%%  dB/dt=bN-λ_1*B+μI-αB-dB
%%%  dI/dt=λ_1*B-μI+αB-λ_2*I-dI
%%%  dR/dt=-κR+λ_2*I 


OEVCALLS=[100];    
OEVDEATHS=[100];
iterjc=['vars'];
eakfjd=['ALL'];
TAUR=[5];
taures=['tau5resamp']; 
wkst=65;
SM=[0];   

ilptimes=20;
for ilp=1:ilptimes

load us_deaths_weekly_2017-18_noseas
load dlycalls17_18_notrend

dcalls=c17_18_dt;
wdeaths=wsa1(1:104);
wdeaths(1)=mean(wdeaths(2:5));wdeaths(102:104)=wdeaths(99:101);

%%%  Shifting the days

dcalls=c17_18_dt;
dcalls=dcalls(8-SM:728-SM);                       
wcalls=squeeze(sum(reshape(dcalls,7,103)))';
ww=(SM*wdeaths(1:103,1)+(7-SM)*wdeaths(2:104,1))/7;
wdeaths=[ww(1,1);ww(1:102,1)];

wkly=[wdeaths wcalls];
wkly=[wkly;wkly(103,:)];

%%%%%%%%%%%%%%%%
%%% Starting from Week 74 and running inference through Week
%%% 104

wklynlin=wkly(wkst+1:104,:);  %% Into wkst+1 week


%%%%  Define the mapping operator H s.t. H*state returns the state-predicted OBS
%%%%  HH is the number of variables and parameters in state space
%%%%  H is only the number of columns being observed

HH=eye(16);
H1=HH(5,:);   %%  observing deaths
H2=HH(6,:);   %%  observing calls


clear So
num_ens=500;
rnd=rand(16,500);

N=327000000;
So(1,:)=N*ones(1,500);
So(3,:)=10e6+round(10e6*rnd(3,:));   %Ideation 
So(4,:)=1500+round(500*rnd(4,:));  %  Cumulative/Effective/Memory of Removed
So(2,:)=N-So(4,:)-So(3,:);
So(5,:)=round(400*rnd(5,:)); %% Instantaneous removed
So(6,:)=round(4000*rnd(6,:)); %% Instantaneous calls
So(7,:)=2.74e-5;  % b birth
So(8,:)=2.7e-5;  %  d death - should reduce by 4e-5*I/N so
                  %  population doesn't take off
So(11,:)=0.0001+0.00015*rnd(9,:);  % beta contagion ideation contact rate w/ideators
So(12,:)=0.3*rnd(10,:);  % epsilon contagion ideation contact rate w/deaths
So(13,:)=0.3*rnd(11,:);  % tau contagion completion contact rate w/deaths

So(9,:)=0.0049;  % mu rate of loss of ideation
So(10,:)=0.00025;  %  alpha rate of gain of ideation
So(14,:)=7.38e-6;  % gamma 'background' rate of I-->R 
So(15,:)=0.0667;  % kappa memory loss rate
So(16,:)=0.000238;

%%% Having initialized, now integrate and update with iterative
%%% cycles, then move to next block of time and repeat

num_times=20;
xprior=NaN(16,num_ens,num_times);
xpost=xprior;
wkcnt=2;
dt=1;
ucnt=0;

for tt = 1:num_times/wkcnt  %% given in blocks of weeks
    obs_var=var(wkly);
    tt
    cntr=0; tol=1;

    while cntr<10 
        cntr=cntr+1;
        if cntr==1
            [tt cntr tol]
        else
            [tt cntr tol]
            [wkly(wkcnt*(tt-1)+tlp,i) prior_mean]
            if isnan(prior_mean)
                'Hohoho'
                pause
            end
        end
        
        %% Random re-initialization
        
        reint=0.05;
        if reint>0
            hmm=ceil(500*rand(500*reint,1));
            rnd=rand(16,500*reint);
            for rit=1:500*reint
                ix=hmm(rit);
                So(1,ix)=N;
                So(3,ix)=10e6+round(10e6*rnd(3,rit));   %Ideation 
                So(4,ix)=1500+round(500*rnd(4,rit));  %  Cumulative/Effective/Memory of Removed
                So(2,ix)=N-So(4,rit)-So(3,rit);
                So(5,ix)=round(400*rnd(5,rit)); %% Instantaneous removed
                So(6,ix)=round(4000*rnd(6,rit)); %% Instantaneous calls
                                                 %So(9,ix)=0.03*rnd(9,rit);
                So(11,ix)=0.015*rnd(9,rit);  % beta contagion ideation contact rate w/ideators
                So(12,ix)=150*rnd(10,rit);  % epsilon contagion ideation contact rate w/deaths
                So(13,ix)=TAUR*rnd(11,rit);  % tau contagion completion contact rate w/deaths
            end
        end
            
        for tlp=1:wkcnt  %%  number of weeks per frame

            for i=1:num_ens
                Sr=[So(1:6,i)]'; %%% Check for aphysicalities
                if Sr(2)/Sr(1)<0.5 | sum(Sr(3:4))>N
                    So(1,i)=N;
                    So(3,i)=10e6+round(10e6*rand(1));   %Ideation 
                    So(4,i)=1500+round(500*rand(1));  %  Cumulative/Effective/Memory of Removed
                    So(2,i)=N-So(4,i)-So(3,i);
                    So(5,i)=round(400*rand(1)); %% Instantaneous removed
                    So(6,i)=round(4000*rand(1)); %% Instantaneous calls
                    So(11,i)=0.015*rand(1);  % beta contagion ideation contact rate w/ideators
                    So(12,i)=150*rand(1);  % epsilon contagion ideation contact rate w/deaths
                    So(13,i)=TAUR*rand(1);  % tau contagion completion contact rate w/deaths
                    Sr=[So(1:6,i)]';
                end
                tcnt=0;
                obstm=7; %%% Run to first observation
                prd=obstm/dt;
                for tt1=dt:dt:obstm
                    tcnt=tcnt+1;
                    
                    p1=poissrnd(dt*So(7,i)*Sr(tcnt,1));  %% bN
                    p2=poissrnd(dt*(So(11,i)*Sr(tcnt,3)+So(12,i)*Sr(tcnt,4))*Sr(tcnt, ...
                                                                      2)/Sr(tcnt,1)); ...
                    %% lambda1*B/N
                    p3=poissrnd(dt*So(9,i)*Sr(tcnt,3));  %% mu*I
                    p4=poissrnd(dt*So(10,i)*Sr(tcnt,2));  %% alpha*B
                    p5=poissrnd(dt*So(8,i)*Sr(tcnt,2));  %% d*B
                    p6=poissrnd(dt*(So(14,i)*Sr(tcnt,3)+So(13,i)*Sr(tcnt,4)*Sr(tcnt,3)/Sr(tcnt,1)));
                    %% lambda2*I/N
                    p7=poissrnd(dt*So(8,i)*Sr(tcnt,3));  %% d*I
                    p8=poissrnd(dt*So(15,i)*Sr(tcnt,4));  %% kappa*R
                    p9=poissrnd(dt*So(16,i)*Sr(tcnt,3));  %% call rate*I
                    
                    bk1=p1+p3-p2-p4-p5;
                    ik1=p2+p4-p3-p6-p7;
                    rk1=p6-p8;
                    nk1=p1-p5-p6-p7;
                    
                    if isnan(bk1) | isnan(ik1) | isnan(rk1) | ...
                            isnan(nk1)
                        So(1,i)=N;
                        So(3,i)=10e6+round(10e6*rand(1));   %Ideation 
                        So(4,i)=1500+round(500*rand(1));  %  Cumulative/Effective/Memory of Removed
                        So(2,i)=N-So(4,i)-So(3,i);
                        So(5,i)=round(400*rand(1)); %% Instantaneous removed
                        So(6,i)=round(4000*rand(1)); %% Instantaneous calls
                                                     %So(9,i)=0.03*rand(1);
                        So(11,i)=0.015*rand(1);  % beta contagion ideation contact rate w/ideators
                        So(12,i)=150*rand(1);  % epsilon contagion ideation contact rate w/deaths
                        So(13,i)=TAUR*rand(1);  % tau contagion completion contact rate w/deaths
                        Sr(tcnt+1,1:6)=So(1:6,i)';
                        
                    else
                    
                        Sr(tcnt+1,1)=max(0,Sr(tcnt,1)+nk1);
                        Sr(tcnt+1,2)=max(0,Sr(tcnt,2)+bk1);
                        Sr(tcnt+1,3)=max(0,Sr(tcnt,3)+ik1);
                        Sr(tcnt+1,4)=max(0,Sr(tcnt,4)+rk1);
                        Sr(tcnt+1,5)=p6;  %%% Instantaneous deaths
                        Sr(tcnt+1,6)=p9;
                    end
                end
                
                %%  Daily
                
                Dr(:,:,i)=Sr(2:prd+1,:);
                
                %%  Sum the observed for the observation period 
                %%  Either mean or most recent for state variables        
                
                for ij=1:6
                    huh=Sr(2:prd+1,ij);
                    if ij<5
                        Xr(ij,i)=mean(huh)';
                    else
                        Xr(ij,i)=sum(huh)';
                    end
                end
                Xr(7:16,i)=So(7:16,i);
            end
            
            %%%  Loop through observations 

            lambda=1.1;

            if rem(cntr,2)==1
            for i=2:-1:1   %%% Loop 

                inflt=ones(1,16);
                xmn=mean(Xr')';
                xmn=xmn.*inflt';
                xinflated=lambda*(Xr-xmn*ones(1,num_ens))+xmn*ones(1,num_ens);
                Xr=xinflated;
                
                if i==2
                    x=Xr;
                    oyy=mean(mean(Xr));
                    xprior(:,:,tt)=x;
                else        
                    x(11:13,:) = x(11:13,:) + dx(11:13,:);
                    x(1:6,:) = x(1:6,:) + dx(1:6,:);
                    
                    %%  Nudge of negatives
                    if min(min(x))<0
                        x2=x;[m,n]=size(x);
                        x2(x2>=0)=0;x2(x2~=0)=1;
                        x3=x;x3(x3<0)=0;
                        xm=mean(x3')'*ones(1,n);
                        x=x3+x2.*rand(m,n).*xm/2;
                    end
                end
                eval(sprintf('H=H%d;',i));
                prior_var = max(1,var(H*x));
                if i==1
                    ovar=obs_var(i)/OEVDEATHS;
                else
                    ovar=obs_var(i)/OEVCALLS;  %% Note indicing of obs in the 'truth'
                end
                post_var = prior_var*ovar/(prior_var+ovar);
                if prior_var==0
                    post_var=0;
                end
                oy=rcond(1/prior_var+1/ovar);

                prior_mean = mean(H*x);
                post_mean = post_var*(prior_mean/prior_var + wklynlin(wkcnt*(tt-1)+tlp,i)/ovar);
                
                %%%% Compute alpha and adjust distribution to conform to posterior moments
                
                alpha = (ovar/(ovar+prior_var)).^0.5;
                dy = post_mean + alpha*((H*x)-prior_mean)-H*x;
                
                %%%  Loop over each state variable
                
                for j=1:size(Xr,1)
                    A=cov(x(j,:),H*x);
                    rr(j)=A(2,1)/prior_var;
                end
                
                dx=rr'*dy;  
                
                if i==1
                    %%%  Get the new ensemble and save prior and
                    %%%  posterior--ONLY NONLINEAR PARAMETERS
                    
                    xnew=x;
                    xnew(11:13,:) = x(11:13,:) + dx(11:13,:);
                    xnew(1:6,:) = x(1:6,:) + dx(1:6,:);
                    tol=abs(prior_mean-post_mean)/prior_mean;
                    
                    %%  Nudge of negatives
                    
                    if min(min(xnew))<0
                        x2=xnew;[m,n]=size(xnew);
                        x2(x2>=0)=0;x2(x2~=0)=1;
                        x3=xnew;x3(x3<0)=0;
                        xm=mean(x3')'*ones(1,n);
                        xnew=x3+x2.*rand(m,n).*xm/2;
                    end
                    stuff3(wkcnt*(tt-1)+tlp,:,cntr)=[mean(H*x) wkly(tt,i) post_mean prior_var ...
                                  max(H*x) ovar];
                end
            end   % End DA loop of 2 observations
        
            else
                for i=1:2  %%% Loop 

                inflt=ones(1,16);
                xmn=mean(Xr')';
                xmn=xmn.*inflt';
                xinflated=lambda*(Xr-xmn*ones(1,num_ens))+xmn*ones(1,num_ens);
                Xr=xinflated;
                
                if i==1
                    x=Xr;
                    oyy=mean(mean(Xr));
                    xprior(:,:,tt)=x;
                else        
                    x(11:13,:) = x(11:13,:) + dx(11:13,:);
                    x(1:6,:) = x(1:6,:) + dx(1:6,:);
                    
                    %%  Nudge of negatives
                    if min(min(x))<0
                        x2=x;[m,n]=size(x);
                        x2(x2>=0)=0;x2(x2~=0)=1;
                        x3=x;x3(x3<0)=0;
                        xm=mean(x3')'*ones(1,n);
                        x=x3+x2.*rand(m,n).*xm/2;
                    end
                end
                eval(sprintf('H=H%d;',i));
                prior_var = max(1,var(H*x));
                if i==1
                    ovar=obs_var(i)/OEVDEATHS;
                else
                    ovar=obs_var(i)/OEVCALLS;  %% Note indicing of obs in the 'truth'
                end
                post_var = prior_var*ovar/(prior_var+ovar);
                if prior_var==0
                    post_var=0;
                end
                oy=rcond(1/prior_var+1/ovar);
                
                prior_mean = mean(H*x);
                post_mean = post_var*(prior_mean/prior_var + wklynlin(wkcnt*(tt-1)+tlp,i)/ovar);
                
                %%%% Compute alpha and adjust distribution to conform to posterior moments
                
                alpha = (ovar/(ovar+prior_var)).^0.5;
                dy = post_mean + alpha*((H*x)-prior_mean)-H*x;
                
                %%%  Loop over each state variable
                
                for j=1:size(Xr,1)
                    A=cov(x(j,:),H*x);
                    rr(j)=A(2,1)/prior_var;
                end
                
                dx=rr'*dy;  
                
                if i==2
                    %%%  Get the new ensemble and save prior and
                    %%%  posterior
                    
                    xnew=x;
                    xnew(11:13,:) = x(11:13,:) + dx(11:13,:);
                    xnew(1:6,:) = x(1:6,:) + dx(1:6,:);
                    tol=abs(prior_mean-post_mean)/prior_mean;
                    
                    %%  Nudge of negatives
                    
                    if min(min(xnew))<0
                        x2=xnew;[m,n]=size(xnew);
                        x2(x2>=0)=0;x2(x2~=0)=1;
                        x3=xnew;x3(x3<0)=0;
                        xm=mean(x3')'*ones(1,n);
                        xnew=x3+x2.*rand(m,n).*xm/2;
                    end
                    stuff3(wkcnt*(tt-1)+tlp,:,cntr)=[mean(H*x) wkly(tt,i) post_mean prior_var ...
                                  max(H*x) ovar];
                end
            end   % End DA loop of 2 observations
         
            end
            
            xpost2(:,:,wkcnt*(tt-1)+tlp,cntr)=xnew;
            xpost(:,:,wkcnt*(tt-1)+tlp)=xnew;
            xpost3(:,:,tlp,tt,cntr)=xnew;
            So=xnew;
            mSo=mean(So')';
            
        end  %%  End of tlp loop -- i.e. a single iteration of a
             %%  time series segment

             
        
        if tt==1  % If this is the first block then re-initialize
                  % state variables to initial state, but adjust
                  % the parameters
            rnd=rand(16,500);
            So(1,:)=N*ones(1,500);
            So(3,:)=10e6+round(10e6*rnd(3,:));   %Ideation 
            So(4,:)=1500+round(500*rnd(4,:));  %  Cumulative/Effective/Memory of Removed
            So(2,:)=N-So(4,1)-So(3,1);
            So(5,:)=round(400*rnd(5,:)); %% Instantaneous removed
            So(6,:)=round(4000*rnd(6,:)); %% Instantaneous calls
            
        else   %%  Otherwise just using posterior of previous week block
            
            rnd=randn(16,500);
            So(1,:)=N*ones(1,500);
            So(3,:)=max(0,mSo(3)+round(5e6.^(0.95^cntr)*rnd(3,:)));   %Ideation 
            So(4,:)=max(0,mSo(4)+round(250.^(0.95^cntr)*rnd(4,:)));  %  Cumulative/Effective/Memory of Removed
            So(2,:)=max(0,N-So(4,:)-So(3,:));
            
        end     
             
             
        for iaph=1:num_ens
            aph=[So(1:6,iaph)]';
            if aph(2)/aph(1)<0.5 | sum(aph(3:4))>N | aph(5)>0.1*N ...
                    | aph(6)>0.1*N
                So(1,iaph)=N;
                So(3,iaph)=10e6+round(10e6*rand(1));   %Ideation 
                So(4,iaph)=1500+round(500*rand(1));  %  Cumulative/Effective/Memory of Removed
                So(2,iaph)=N-So(4,iaph)-So(3,iaph);
                So(5,iaph)=round(400*rand(1)); %% Instantaneous removed
                So(6,iaph)=round(4000*rand(1)); %% Instantaneous calls
                So(11,iaph)=0.015*rand(1);  % beta contagion ideation contact rate w/ideators
                So(12,iaph)=150*rand(1);  % epsilon contagion ideation contact rate w/deaths
                So(13,iaph)=TAUR*rand(1);  % tau contagion completion contact rate w/deaths
            end
        end
 
        [mSo(1:6) mean(So(1:6,:)')' max(So(1:6,:)')' min(So(1:6,:)')']

    end  %%% End of 10 iterations

    
    if tt>num_times
        tSo=xnew(1:6,:);
        mSo=mean(tSo')';
        rndn=randn(16,500);
        rnd=rand(16,500);
        So(1,:)=N*ones(1,500);
        So(3,:)=max(0,mSo(3)+round(5e6*rndn(3,:)));   %Ideation 
        So(4,:)=max(0,mSo(4)+round(250*rndn(4,:)));  %  Cumulative/Effective/Memory of Removed
        So(2,:)=max(0,N-So(4,:)-So(3,:));
        So(5,:)=round(400*rnd(5,:)); %% Instantaneous removed
        So(6,:)=round(4000*rnd(6,:)); %% Instantaneous calls
        So(11,:)=0.015*rnd(11,:);  % beta contagion ideation contact rate w/ideators
        So(12,:)=150*rnd(12,:);  % epsilon contagion ideation contact rate w/deaths
        So(13,:)=TAUR*rnd(13,:);  % tau contagion completion contact rate w/deaths
    end

end        
            
Xpost(ilp,:,:,:)=shiftdim(xpost,1);
            
save KSAB_super_ens_Xpost_mu0p0049_finalfig4 Xpost
ilp
if ilp<ilptimes
    clearvars -except Xpost ilptimes ilp OEVCALLS OEVDEATHS iterjc eakfjd TAUR taures wkst SM 
end

end 

Xpost=reshape(Xpost,ilp*500,20,16);

truth=squeeze(wklynlin);
time=1:num_times;

mpost=squeeze(mean(Xpost));
mprior=squeeze(mean(shiftdim(xprior,1)));
spost=squeeze(std(Xpost));
lbpost=mpost-2*spost;
ubpost=mpost+2*spost;

%%%  Final Form Figures

dt=datenum('jan-8-2017'); dts=dt+(wkst)*7-SM; %% Synching date CORRECTLY to beginning of week
tmm=dts+6:7:dts+6+7*19; %% +6 Puts to week ENDING DATE
dtss=tmm(1);
figure(20)
set(gcf,'PaperPosition',[0.5 0.5 7.5 10])
set(gca,'FontSize',14);
set(gca,'linewidth',1.5)
set(gca,'TickDir','out');
set(gca,'Ticklength',[0.01,0.01]);
xtickangle(0)
box off
colors=[215,48,39,255; 252,141,89,255; 145,191,219,255; 69,117,180,255]/255;
for i=1:3
subplot(3,1,i)
plot(tmm,mpost(:,i+10),'Linewidth',1.5,'Color',colors(i,:))
hold on
plot(tmm,lbpost(:,i+10),'--','Linewidth',1.5,'Color',colors(i,:))
plot(tmm,ubpost(:,i+10),'--','Linewidth',1.5,'Color',colors(i,:))
xlim([min(tmm) max(tmm)])
tickLocations = datenum([dtss+14 dtss+42 dtss+70 dtss+98 dtss+126]);
set(gca,'XTick',tickLocations)
datetick('x','mmm dd, yyyy','keepticks','keeplimits')
hold off
oyu=[ubpost(:,i+10)]; oyu=max(max(oyu)); oyu=1.2*oyu;
if i==1
    title('\beta - Baseline Contact with Ideators')    
    axis([min(tmm) max(tmm) 0 oyu])
elseif i==2
    title('\epsilon - Baseline Contact with Memory of Deaths')
    axis([min(tmm) max(tmm) 0 oyu])
else
    title('\tau - Ideator Contact with Memory of Deaths')
    axis([min(tmm) max(tmm) 0 oyu])
end
legend('Posterior','95% CI','Location','Northwest')
legend boxoff
set(gca,'FontSize',14);
set(gca,'linewidth',1.5)
set(gca,'TickDir','out');
set(gca,'Ticklength',[0.01,0.01]);
xtickangle(0)
box off
end

savefig('mainfigure4B.fig')
saveas(gcf,'mainfigure4B','epsc')
saveas(gcf,'mainfigure4B','png') 
saveas(gcf,'mainfigure4B','pdf')


dt=datenum('jan-8-2017'); dts=dt+(wkst)*7-SM; %% Synching date CORRECTLY to beginning of week
tmm=dts+6:7:dts+6+7*19; %% +6 Puts label to week ENDING DATE 
dtss=tmm(1);
figure(21)
set(gcf,'PaperPosition',[0.5 0.5 7.5 10])
set(gca,'FontSize',14);
set(gca,'linewidth',1.5)
set(gca,'TickDir','out');
set(gca,'Ticklength',[0.01,0.01]);
xtickangle(0)
box off
colors=[215,48,39,255; 252,141,89,255; 145,191,219,255; 69,117,180,255]/255;
for i=1:4
subplot(4,1,i)
plot(tmm,mpost(:,i+2),'b','Linewidth',1.5,'Color',colors(i,:)); hold on

plot(tmm,lbpost(:,i+2),'--b','Linewidth',1.5,'Color',colors(i,:))
if i==3
        plot(tmm,squeeze(truth(1:num_times,1)),'linestyle','none', ...
             'marker','o','MarkerFaceColor','k')
elseif i==4
    plot(tmm,squeeze(truth(1:num_times,2)),'linestyle','none','marker','o','MarkerFaceColor','k')
end
plot(tmm,ubpost(:,i+2),'--b','Linewidth',1.5,'Color',colors(i,:))
hold off
xlim([min(tmm) max(tmm)])
tickLocations = datenum([dtss+14 dtss+42 dtss+70 dtss+98 dtss+126]);
set(gca,'XTick',tickLocations)
datetick('x','mmm dd, yyyy','keepticks','keeplimits')
if i==1
    title('Ideation')
elseif i==2
    title('Memory of Removed')
elseif i==3
    title('Weekly Deaths')
else
    title('Weekly Calls')
end
if i>2
legend('Posterior','95% CI','Truth','Location','Northwest')
legend boxoff
else 
    legend('Posterior','95% CI','Location','Northwest')
legend boxoff
end
set(gca,'FontSize',14);
set(gca,'linewidth',1.5)
set(gca,'TickDir','out');
set(gca,'Ticklength',[0.01,0.01]);
xtickangle(0)
box off
end

