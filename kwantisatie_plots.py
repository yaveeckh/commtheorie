def run_kwantisatie():
    obj = Kwantisatie(0)
    
    ########################
    # LINEAIRE KWANTISATOR #
    ########################

    # Maak een figuur van de optimale SQR voor lineaire kwantisatie in dB 
    # in functie van alpha = [2,...,8], waarbij M = 2**alpha
    
    """
    print('Generating plot: optimal SQR in function of alpha..')
    alpha = np.array([i for i in range(2,9)])
    y = np.array([obj.bepaal_optimale_lineaire_kwantisator(2**i)[2] for i in range(2,9)])
    y2 = np.array([obj.bepaal_compansie_kwantisator(2**i)[1] for i in range(2,9)])
    y3 = np.array([obj.bepaal_Lloyd_Max_kwantisator(2**i)[1] for i in range(2,7)])
    winst = [0 for _ in range(0,6)] 
    for i in range(0,6):
        winst[i] = y[i+1] - y[i]
    print(winst)
    plt.plot(alpha, y)
    plt.plot(alpha, y2)
    plt.plot(alpha[:5], y3)
    plt.xlabel("Alpha")
    plt.ylabel("SQR [dB]")
    plt.savefig('SQR.png')
    plt.close()
    print('Done!')
    """
    
    # opt_lin_kwant = obj.bepaal_optimale_lineaire_kwantisator(2**6, True)
    # r_opt_lin = opt_lin_kwant[4]
    # q_opt_lin = opt_lin_kwant[5]
    # gekwantiseerd_lin = obj.kwantiseer(r_opt_lin, q_opt_lin)
    
    
    """
    print('Generating plot: fU(u)')
    plt.figure(figsize=(20,10))
    for i in range(0, 2**6):
        plt.axvline(q_opt_lin[i], 0, 0.1, color = 'k', lw = 0.5)
        plt.axvline(r_opt_lin[i], 0, 0.2, color = 'r', lw = 0.5)
    
    plt.xlabel("Monsterwaarde u")
    plt.ylabel("dichtheid")
    obj.plot_distributie('fu_uniform.png')
    print('Done!')
    """

    #########################
    # COMPANSIE KWANTISATOR #
    #########################
    
    # compansie_kwant = obj.bepaal_compansie_kwantisator(2**6)
    # r_compansie = compansie_kwant[3]
    # q_compansie = compansie_kwant[4]
    # gekwantiseerd_compansie = obj.kwantiseer(r_compansie, q_compansie)

    """
    print('Generating plot: fU(u)')
    plt.figure(figsize=(20,10))
    for i in range(0, 2**6):
        plt.axvline(q_compansie[i], 0, 0.1, color = 'k', lw = 0.5)
        plt.axvline(r_compansie[i], 0, 0.2, color = 'r', lw = 0.5)
    plt.axvline(r_compansie[2**6], 0, 0.2, color = 'r', lw = 0.5)
    plt.xlabel("Monsterwaarde u")
    plt.ylabel("dichtheid")
    obj.plot_distributie('fu_compansie.png')
    print('Done!')
    """

    
    #########################
    # Lloyd-Max KWANTISATOR #
    #########################

    opt_kwant = obj.bepaal_Lloyd_Max_kwantisator(2**6)
    r_opt = opt_kwant[3]
    q_opt = opt_kwant[4]
    gekwantiseerd_opt = obj.kwantiseer(r_opt, q_opt)

    """
    print('Generating plot: fU(u)')
    plt.figure(figsize=(20,10))
    for i in range(0, 2**6):
        plt.axvline(q_opt[i], 0, 0.1, color = 'k', lw = 0.5)
        plt.axvline(r_opt[i], 0, 0.2, color = 'r', lw = 0.5)
    plt.axvline(r_opt[2**6], 0, 0.2, color = 'r', lw = 0.5)
    plt.xlabel("Monsterwaarde u")
    plt.ylabel("dichtheid")
    obj.plot_distributie('fu_opt.png')
    print('Done!')
    """

    ###########################

    # Sla de gekwantiseerde fragmenten ook op
    #obj.save_and_play_music(obj.kwantiseer(r_opt_lin, q_opt_lin), "uniform.wav", 0)
    #obj.save_and_play_music(obj.kwantiseer(r_compansie, q_compansie), "compansie.wav", 0)
    #obj.save_and_play_music(np.array(obj.kwantiseer(r_opt, q_opt)), "LM.wav", 0)
    
    return (r_opt,q_opt,gekwantiseerd_opt)
