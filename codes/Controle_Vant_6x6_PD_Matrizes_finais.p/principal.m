function [xs,phi_des,theta_des,tau,erro_total]=principal(dt,t,xs,m,Ixx,Iyy,Izz,g,x_des,y_des,z_des,x_dot_des,y_dot_des,z_dot_des)
tau=[m*g;0;0;0]; erro_total=[0;0;0;0;0;0];
for ii=2:length(t)
    
    M_eta = matmassa(xs(:,ii-1),m,Ixx,Iyy,Izz);                                                    % Matriz de massa
    C_eta = coriolis(xs(:,ii-1),m,Ixx,Iyy,Izz);                                                    % Matriz de Coriolis
    B_eta = entradaB(xs(:,ii-1));                                                                  % Vetor de entrada
    G_eta=[0,0,m*g,0,0,0]';                                                                        % Vetor de Gravidade
       
    % Controle Proporcional Derivativo - PD
    
    [U1,phi_des0,theta_des0] = outerloop(xs(:,ii-1),x_des(ii-1),y_des(ii-1),z_des(ii-1),x_dot_des(ii-1),y_dot_des(ii-1),z_dot_des(ii-1),m,g); 
    
    phi_des(ii)=phi_des0; theta_des(ii)=theta_des0;
    
    [U2,U3,U4] = innerloop(xs(:,ii-1),phi_des(ii-1),theta_des(ii-1));
    
        errox(ii-1)=x_des(ii-1)-xs(7,ii-1);
        erroy(ii-1)=y_des(ii-1)-xs(8,ii-1);
        erroz(ii-1)=z_des(ii-1)-xs(9,ii-1);
        errophi(ii-1)=phi_des(ii-1)-xs(10,ii-1);
        errotheta(ii-1)=theta_des(ii-1)-xs(11,ii-1);
        erropsi(ii-1)=0-xs(12,ii-1);
        
        erro_total(:,ii)=[errox(ii-1),erroy(ii-1),erroz(ii-1),errophi(ii-1),errotheta(ii-1),erropsi(ii-1)]';
  
        tau(:,ii) = [U1;U2;U3;U4];
        u(:,ii) = [tau(:,ii-1);zeros(8,1)];
        
        % Espaço de estados x'=Ax+Bu+Xog
        
        A = [-inv(M_eta)*C_eta,zeros(6,6);eye(6,6),zeros(6,6)];
        
        B = [inv(M_eta)*B_eta,zeros(6,8);zeros(6,12)];
        
        Xog = [-inv(M_eta)*G_eta;zeros(6,1)];

        k1=A*xs(:,ii-1)               +  B*u(:,ii) + Xog;
        k2=A*(xs(:,ii-1) + 0.5*k1*dt) +  B*u(:,ii) + Xog;
        k3=A*(xs(:,ii-1) + 0.5*k2*dt) +  B*u(:,ii) + Xog;
        k4=A*(xs(:,ii-1) + k3*dt)     +  B*u(:,ii) + Xog;
        
        xs(:,ii)=xs(:,ii-1)+(dt/6)*(k1+2*k2+2*k3+k4);

end

end