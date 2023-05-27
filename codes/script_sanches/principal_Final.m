%% Main script - Equa��es do movimento para um VANT sem carga - 6x6 PD
% Renan Sanches Geronel

clear all; clc; 

% Controle utilizado PD https://www.cambridge.org/core/journals/aeronautical-journal/article/abs/dynamic-responses-due-to-the-dryden-gust-of-an-autonomous-quadrotor-uav-carrying-a-payload/07E3CD5EC5B160FFE51BFA2AC4176114

%% Condi��es iniciais
nnn = 1000;
tau_all = cell(nnn,1);
xs_all = cell(nnn,1);

  for loop=1:1000
      
    t0=0;                                                                   % Tempo inicial
    tf=40;                                                                  % Tempo final
    dt=0.002;                                                               % Discretiza��o do Tempo
    Fs=1/dt;                                                                % Frequ�ncia de amostragem
    t=t0:dt:tf;                                                             % Vetor tempo
    
    [xs,m,l,lambda,Ixx,Iyy,Izz,g,M,kf,km,Omegar,Cdj,Aj]=parametros;         % Par�metros do Quad (massa,in�rcia e geom�tricos)
    option = 2;         % 1 = retangular  2 = circular  e 3 = linear        % Escolher a trajet�ria desejada
    
    x_des(1)=0;y_des(1)=0;z_des(1)=0; x_dot_des(1)=0;y_dot_des(1)=0;z_dot_des(1)=0;     % Condi��o inicial das trajet�rias
    for ii=1:length(t)
    [x_des0,y_des0,z_des0,x_dot_des0,y_dot_des0,z_dot_des0]=trajetoria(t(ii),option, loop);                        % Trajet�rias desejadas
    x_des(ii)=x_des0;y_des(ii)=y_des0;z_des(ii)=z_des0;x_dot_des(ii)=x_dot_des0;y_dot_des(ii)=y_dot_des0;z_dot_des(ii)=z_dot_des0;
    end
    
    % Criar pcode .p - proteger arquivo! https://www.mathworks.com/help/matlab/ref/pcode.html
    
 [xs,phi_des,theta_des,tau,erro_total]=principal(dt,t,xs,m,Ixx,Iyy,Izz,g,x_des,y_des,z_des,x_dot_des,y_dot_des,z_dot_des);
 
 % xs = vetor de estados, phi_des e theta_des = angulos desejados, tau = [U1,U2,U3,U4] e erro_total = erros entre (xs_des-xs)
 % xs Conferir abaixo. 
 % O vetor xs gera respectivamente (12 estados) [x' y' z' phi' theta' psi' x y z phi theta psi];
 %note que x' eh a derivada de x, sendo x deslocamento e x' velocidade
 
 % Funcao nftool
    % str = compose("xs_%1d.csv",loop);
    % csvwrite(str, xs)
    % str2 = compose("tau_%1d.csv",loop);
    % csvwrite(str2, tau)

    tau_all{loop} = tau;
    xs_all{loop} = xs;
    disp(loop)

  end
 %% Figuras

% figure();
% plot3(x_des(1:end-2),y_des(1:end-2),z_des(1:end-2),'r', 'linewidth',1.6);      
% hold on;
% plot3(xs(7,1:end-2),xs(8,1:end-2),xs(9,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$x(t)$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$y(t)$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  zlabel('$z(t)$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% 
% figure();
% subplot(3,1,1);
% plot(t(1:end-2),x_des(1:end-2),'r', 'linewidth',1.6);  
% hold on;
% plot(t(1:end-2),xs(7,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('${x}(t)$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% 
% subplot(3,1,2);
% plot(t(1:end-2),y_des(1:end-2),'r', 'linewidth',1.6);  
% hold on;
% plot(t(1:end-2),xs(8,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('${y}(t)$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% 
% subplot(3,1,3);
% plot(t(1:end-2),z_des(1:end-2),'r', 'linewidth',1.6);  
% hold on;
% plot(t(1:end-2),xs(9,1:end-2),'-b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
% ylim([0 4]);
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('${z}(t)$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% 
% figure();
% subplot(3,1,1);
% plot(t(1:end-2),phi_des(1:end-2),'r', 'linewidth',1.6);  
% hold on;
% plot(t(1:end-2),xs(10,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$\phi$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% subplot(3,1,2);
% plot(t(1:end-2),theta_des(1:end-2),'r', 'linewidth',1.6);  
% hold on;
% plot(t(1:end-2),xs(11,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$\theta$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% subplot(3,1,3);
% plot(t(1:end-2),xs(12,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$\psi$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% 
% figure();
% subplot(2,2,1);plot(t(1:end-2),tau(1,1:end-2),'b', 'linewidth',1.6);             
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$U_1$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% subplot(2,2,2);
% plot(t(1:end-2),tau(2,1:end-2),'b', 'linewidth',1.6);              
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$U_2$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% subplot(2,2,3);
% plot(t(1:end-2),tau(3,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$U_3$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
% subplot(2,2,4);
% plot(t(1:end-2),tau(4,1:end-2),'b', 'linewidth',1.6);                
% grid on;  
% font_size=12;
%  xlabel('$t$','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
%  ylabel('$U_4$ ','FontUnits','points','interpreter','latex',...
% 'FontSize',font_size,'FontName','Times');
% set(gca,'Units','normalized','FontUnits','points','FontWeight','normal',...
%     'FontSize',font_size,'FontName','Times');
    