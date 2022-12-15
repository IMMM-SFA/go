import numpy as np


def tinney_one(erp, piv_ind, piv_ord, ex_bus):
    """Subroutine TinneyOne applies Tinney 1 optimal ordering to the input data
    in order to reduce the fills generated in the partial LU factorization
    process.

    Parameters
    ----------
    erp: 1*n array
        Includes end of row pointer of input addmittance matrix
    piv_ind: 1*n array
        Includes bus ordering after pivotting
    piv_ord: 1*n array
        Includes bus indices after pivotting
    ex_bus: 1*n array
        Includes bus indices of external buses

    Returns
    -------
    piv_ind: 1*n array
        Includes bus ordering after pivotting by Tinney 1 ordering
    piv_ord: 1*n array
        Includes bus indices after pivotting by Tinney 1 ordering

    Note
    ----
    This subroutine does not pivot any data but output an array includes
    ordering of buses. Pivoting will be done in the subroutine PivotData.

    """

    # Extract the external bus part
    ex_len = len(ex_bus)
    erp_e = erp[0:ex_len+1]

    # Calculate the number of non-zero entry in each row and sort them
    row_len = erp_e[1:] - erp_e[:-1]
    row_len, row_ord = np.sort(row_len)

    # Calculate RowOrdO
    row_ord_o = np.argsort(row_ord)

    # Calculate PivInd and PivOrd
    piv_ind[0:ex_len] = piv_ind[row_ord_o]
    piv_ord[piv_ind[:ex_len]] = np.arange(ex_len)

    return piv_ord, piv_ind


def pivot_data(data_b, erp, c_indx, ex_bus, numb, bound_bus):
    """Subroutine PivotData do pivotting to the input addmittance matrix. Two
    pivotting will be done: 1. columns and rows corresponding to external
    buses will be pivotted to the top left corner of the input matrix. 2.
    Tinney One optimal ordering strategy will be applied to pivot the data in
    order to reduce fills during (Partial) LU factorization.

    INPUT DATA:
      DataB: 1*n array, includes addmittance data of the full model before
      pivotting
      ERP: 1*n array, includes end of row pointer before pivotting
      CIndx: 1*n array, includes column index pointer before pivotting
      ExBus: 1*n array, includes external bus indices in internal numbering
      NUMB: 1*n array, includes bus numbers in internal numbering

    OUTPUT DATA:
      DataB: 1*n array, includes pivotted addmittance data of the full model
      ERP: 1*n array, includes end of row pointer before pivotting
      CIndx: 1*n array, includes column index pointer
      PivOrd: 1*n array, includes bus indices after pivotting
      PivInd: 1*n array, includes bus ordering after pivotting
    """

    data_bo = np.zeros(data_b.shape)
    c_indx_o = np.zeros(c_indx.shape)
    erp_o = np.zeros(erp.shape)

    # do pivot
    ex_bus = np.sort(ex_bus)
    tf1 = np.isin(numb, ex_bus)
    tf2 = np.isin(numb, bound_bus)
    piv_ind = np.concatenate((np.sort(numb[tf1 == 1]),
                              np.sort(numb[tf2 == 1]),
                              np.sort(numb[(tf1 == 0) & (tf2 == 0)])), axis=None)
    piv_ord = np.zeros(piv_ind.shape)

    for i in range(len(piv_ind)):
        piv_ord[piv_ind[i]] = i

    # Do Tinnney One ordering to reduce fills
    piv_ord, piv_ind = tinney_one(erp, piv_ind, piv_ord, ex_bus)

    # Generate the datas in compact storage format
    for i in range(len(numb)):
        len_ = erp[piv_ind[i] + 1] - erp[piv_ind[i]]
        erp_o[i + 1] = erp_o[i] + len_
        c_indx_o[erp_o[i] + 1:erp_o[i + 1]] = piv_ord[c_indx[erp[piv_ind[i]] + 1:erp[piv_ind[i] + 1]]]
        data_bo[erp_o[i] + 1:erp_o[i + 1]] = data_b[erp[piv_ind[i]] +  1:erp[piv_ind[i] + 1]]

        # Generate the output data
        data_b = data_bo
        c_indx = c_indx_o
        erp = erp_o

    return data_b, erp, c_indx, piv_ord, piv_ind


def self_link(link, start):
    """Subroutine SelfLink search the Link array in by self referencing.
    The searching process will stop until a zero is found. Every non-zero
    element found in the searching process will be stored in LinkArray and
    their corresponding index pointers will be stored in LinkPos. Counter
    will count the number of numzero element in LinkArray.

    INPUT DATA:
        Link - N * 1 array containing the link list
        Start- scalar is the starting point of the searching process

    OUTPUT DATA:
        Counter - scalar, number of non-zero elements in LinkArray
        LinkArray - N*1 array, containing the non-zero element found in the
            searching process
        LinkPos - N*1 array, containing the index pointers of the non-zero
            elements stored in LinkArray

    INTERNAL DATA:
        SelRef - scalar, used to do searching in Link list, which is equal to
            the current found element and also pointer to the next element

    NOTE:
        1. Initiate the SelfRef=Start and Counter=0. Go to step 2.
        2. While the element found in the Link list (Link(SelfRef)) is not
            zero go to step 2.1 if equal to zero go to step 3.
            2.1 Increment Counter by 1. Go to step 2.2
            2.2 Store SelfRef to LinkPos which is the index of current
                non-zero element. Go to step 2.3
            2.3 Update the SelfRef equal to Link(SelfRef). Go to step 2.4
            2.4 Store the SelfRef value to LinkArray which is the value of
                current element. Go to step 2.
        3. End of the subroutine. Return.
    """

    self_ref = start
    counter = 0
    link_pos = []
    link_array = []
    while link[self_ref] != 0:
        counter += 1
        link_pos.append(self_ref)
        self_ref = link[self_ref]
        link_array.append(self_ref)

    return np.array(link_pos), np.array(link_array), counter


def rod_assignment(erp, c_indx, c_indx_u, erp_u, min_nod, switch, self_ref, row_index):
    """Subroutine RODAssignment is called in the symbolic LU factorization to
    assign column index of non-zero element on right of the diagonal in U matrix.
    The non-zero element may come be native or filed.

    Parameters
    ----------
    ERP : (N_node+1)*1 array
        Containning the end of row pointer data
    CIndx : N*1 array
        Containning the column index of the rows, N is the number of
        non-zeros elements in the input data
    CindxU : N*1 array
        Containning the column index of rows in the U matrix. The length N
        depends on the number of non-zero elements in previous rows on right
        of diagonal.
    Switch : N*1 array
        Used to record the index off diagonal element in CIndxU and avoid
        keep the indices disjoint
    SelfRef : scalar
        Data value in the self refertial link also is the pointer point to
        next position in the self referential link
    RowIndex : scalar
        The index of the row in processing
    ERPU : N_dim*1 array
        Containing end of row pointer of all rows except the last row which
        doesn have any off diagonal element, N_dim is the dimension of the
        input matrix A in the original Ax = b problem
    MinNod : scalar
        Store the minimum index of non-zero element on the right of the
        diagonal

    Returns
    -------
    ERPU : N_dim*1 array
        Containing end of row pointer of all rows except the last row which
        doesn have any off diagonal element, N_dim is the dimension of the
        input matrix A in the original Ax = b problem
    CIndxU : N*1 array
        Containing the column index of off diagonal element in each row
        in the U matrix. The length N depends on the number of native plus
        filled non-zero elements in the off diagonal position in U matrix.
        CIndxU is ordered.
    MinNod : scalar
        Store the minimum index of non-zero element on the right of the
        diagonal
    Switch : N*1 array
        Used to record the index off diagonal element in CIndxU and avoid
        keep the indices disjoint

    Note
    ----
    In output data, ERPU, CIndxU, MinNod and Switch are updated in the
    subroutine. The output will return the updated arrays.
    """

    row_len = erp[self_ref+1] - erp[self_ref]  # number of non-zero element in current row
    row_col_ind = c_indx[erp[self_ref]+1:erp[self_ref+1]]

    # In the loop every time read one ROD element ERPU(RowColInd)+1
    if erp_u[row_index+1] == 0:
        erp_u[row_index+1] = erp_u[row_index]

    for i in range(row_len): # dealing with each non-zero native non-zero element first
        # Load the native non-zero ROD
        if row_col_ind[i] > row_index and (switch[row_col_ind[i]]!=row_index):
            c_indx_u[erp_u[row_index+1]+1] = row_col_ind[i]
            erp_u[row_index+1] += 1

            # Update the MinNod if MinNod greater than current column index RowColInd[i]
            min_nod = min(min_nod, row_col_ind[i])
            # Update the Switch list
            switch[row_col_ind[i]] = row_index

    return erp_u, c_indx_u, min_nod, switch


def pre_process_data(mpc, ex_bus):
    """PreProcessData(mpc, ExBus) performs a series of tasks on an input model to
    update and clean it. This includes:
    1. Eliminating all isolated buses
    2. Eliminating all out-of-service branches
    3. Eliminating all in-service but connected to isolated bus branches
    4. Eliminating all HVDC line connected to isolated buses
    5. Eliminating all generators on isolated buses
    6. Updating the list of external buses (ExBus) by eliminating the isolated
    buses in the list

    Parameters
    ----------
    mpc : struct
        Input original full model (MATPOWER case file).
    ExBus : 1*n array
        Original list of external buses.

    Returns
    -------
    mpc : struct
        Updated model.
    ExBus : 1*n array
        Updated list of external buses.
    """

    mpc = np.sort(np.array(mpc), axis=(0,1))
    numbr = mpc.shape[0]

    # Eliminate all out-of-service lines
    mpc = mpc[mpc[:, 11] != 0]

    # Find isolated buses
    isobus = mpc[mpc[:, 2] == 4, 1]
    print(f"Eliminate {len(isobus)} isolated buses")

    # Eliminate branches connected to isolated buses
    mpc = mpc[~(np.isin(mpc[:, 1], isobus) | np.isin(mpc[:, 2], isobus))]
    print(f"Eliminate {numbr - mpc.shape[0]} branches")

    # Eliminate isolated buses
    mpc = mpc[mpc[:, 2] != 4]

    # Eliminate all generators on isolated buses
    mpc = mpc[~np.isin(mpc[:, 1], isobus)]
    print(f"Eliminate {len(np.where(np.isin(mpc[:, 1], isobus)))} generators")

    # Update external bus list
    ex_bus = ex_bus[~np.isin(ex_bus, isobus)]

    # If HVDC lines exist, eliminate HVDC lines connected to isolated buses
    if 'dcline' in mpc.keys():
        mpc = mpc[~(np.isin(mpc[:, 1], isobus) | np.isin(mpc[:, 2], isobus))]
        print(f"Eliminate {len(np.where(np.isin(mpc[:, 1], isobus) | np.isin(mpc[:, 2], isobus)))} dc lines")

    print("Preprocessing complete")

    return mpc, ex_bus


def eq_rod_assignment(erp, c_indx, cindxu, erpu, min_nod, switch, self_ref, row_index, chain, min_nod1):
    """Subroutine EQRODAssignment is called in symbolic LU factorization, this
    subroutine is specifically used to identify the pointers of equivalent
    branches spanning the boundary buses. The equivalent branches are the
    fills in the factorization process of rows and columns of the external
    buses.

    INPUT DATA:
      erp - (N_node+1)*1 array containning the end of row pointer data
      c_indx - N*1 array containning the column index of the rows, N is the
               number of non-zeros elements in the input data
      cindxu- N*1 array containning the column index of rows in the U matrix.
               The length N dedpends on the number of non-zero elements in
               previous rows on right of diagonal.
      erpu -  N_dim*1 array containing end of row pointer of all rows except
              the last row which doesn have any off diagonal element, N_dim is
              the dimension of the input matrix A in the original Ax = b
              problem
      switch- N*1 array which is used to record the index off diagonal
              element in CIndxU and avoid keep the indices disjoint
      self_ref-scalar, data value in the self refertial link also is the pointer
               point to next position in the self referential link
      row_index-scalar, the index of the row in processing
      chain- 1*n array, if the MinNod got from this cycle (MinNod1) is
           different to the previous one and the previous one is not inf, then Chain(MinNod1) will
           record the value of Link(MinNod1)
      min_nod -scalar, store the minimum index of non-zero element on the
               right of the diagonal
      min_nod1 -scalar, MinNod value recorded in the last cycle

    OUTPUT DATA:
      erpu -   N_dim*1 array containing end of row pointer of all rows except
               the last row which doesn have any off diagonal element, N_dim is
               the dimension of the input matrix A in the original Ax = b
               problem
      cindxu - N*1 array containing the column index of off diagonal element
               in each row in the U matrix. The length N depends on the
               number of native plus filled non-zero elements in the off
               diagonal position in U matrix. CIndxU is ordered.
      min_nod - scalar, store the minimum index of non-zero element on the
               right of the diagonal
      switch- N*1 array which is used to record the index off diagonal
              element in CIndxU and avoid keep the indices disjoint
      chain_flag- scalar, indicate if the MinNod is changed
    NOTE: in output data, erpu, cindxu, min_nod and switch are updated in the
    subroutine. The output will return the updated arrays.

    NOTE: This subroutine is only used to generate the equivalent line
    indices.
    """

    # Number of non-zero element in current row
    row_len = erp[self_ref+1] - erp[self_ref]
    row_col_ind = c_indx[erp[self_ref]+1:erp[self_ref+1]]

    # Initiate the ERPU to be the end of last row
    if erpu[row_index+1] == 0:
        erpu[row_index+1] = erpu[row_index]  # In the loop every time read one ROD element ERPU(RowColInd)+1
    chain_flag = 0
    row_col_ind2 = row_col_ind[row_col_ind > row_index]
    for i in range(row_len):  # dealing with each non-zero native non-zero element first
        if row_col_ind[i] > row_index and switch[row_col_ind[i]] != row_index:  # check if current element is on ROD
            cindxu[erpu[row_index+1]+1] = row_col_ind[i]
            erpu[row_index+1] = erpu[row_index+1] + 1  # increase 1 after reading one non-zero number;
            min_nod = min(min_nod, row_col_ind[i])  # Update the MinNod if MinNod greater than current column index RowColInd(i)
            switch[row_col_ind[i]] = row_index  # Update the Switch list
        elif chain[row_col_ind[i]] != 0 and row_col_ind[i] > row_index:
            min_nod = min(min_nod, row_col_ind[i])  # Update the MinNod
    if len(row_col_ind2) > 0 and np.min(row_col_ind2) == min_nod1 and min_nod != min_nod1:
        chain_flag = 1
        min_nod = np.inf
    return cindxu, erpu, min_nod, switch, chain_flag


def partial_sym_lu(c_indx, erp, dim, stop, bound_bus):
    """Subroutine PartialSymLU do partial symbolic LU factorization.

    Parameters
    ----------
    c_indx : array
        N*1 array containing the column index of the rows, N is the
        number of non-zeros elements in the input data
    erp : array
        (N_node+1)*1 array containing the end of row pointer data
    dim : int
        scalar, dimension of the input matrix
    stop : int
        scalar, stop sign of the LU factorization (The LU factorization
        in the reduction process is not complete but partial)
    bound_bus : array
        1*n array, includes indices of boundary buses

    Returns
    -------
    erp_u : array
        N_dim*1 array containing end of row pointer of all rows except
        the last row which doesnt have any off diagonal element, N_dim is
        the dimension of the input matrix A in the original Ax = b
        problem
    c_indx_u : array
        N*1 array containing the column index of off diagonal element
        in each row in the U matrix. The length N depends on the
        number of native plus filled non-zero elements in the off
        diagonal position in U matrix. CIndxU is unordered.
    erp_eq : array
        N*1 arrays, together include the indices of
        equivalent branches
    c_indx_eq : array
        N*1 arrays, together include the indices of
        equivalent branches
    """

    num_row = dim # number of rows of given data matrix
    # num_col = dim # number of columns of given data matrix

    # Initialization
    c_indx_u = 0
    erp_u = np.zeros(dim) # with additional one digit for the 7th row
    switch = np.zeros(dim)
    min_nod_0 = float('inf') # This is a initiate large value of MinNod; not the MinNod used in building the symbolic structure
    min_nod = min_nod_0
    min_nod_1 = min_nod
    link = erp_u
    c_indx_eq = 0
    erp_eq = np.zeros(stop + len(bound_bus) + 1)
    chain = np.zeros(stop + len(bound_bus))

    # preprocess the data by ordering the CIndx of every row in ascending order
    for i in range(1, num_row + 1):
        row_col_ind = np.arange(erp[i - 1], erp[i]) # for every row the pointer of the column index
        c_indx[row_col_ind] = np.sort(c_indx[row_col_ind])

    # initiate starting row 1
    row_index = 1
    while row_index <= len(bound_bus) + stop:
        min_nod_1 = min_nod_0

        # Step 1
        if row_index <= stop:
            c_indx_u, erp_u, min_nod, switch = rod_assignment(erp, c_indx, c_indx_u, erp_u, min_nod, switch, row_index, row_index)

            # Step 2
            # Check fill in ROD in current row
            self_ref = row_index # self referential link pointer to next available link element
            while (link[self_ref] != 0):
                self_ref = link[self_ref] # to refer for the next link element
                c_indx_u, erp_u, min_nod, switch = rod_assignment(erp_u, c_indx_u, c_indx_u, erp_u, min_nod, switch, self_ref, row_index)
            link[row_index] = 0 # zero the element in the Link list of current row

            # Step 3
            # update Link node (MinNod) to be current RowIndex
            if min_nod > row_index and min_nod != min_nod_0:
                self_ref = min_nod # start the assign value to Link
                while (link[self_ref] != 0):
                    self_ref = link[self_ref]
                link[self_ref] = row_index
            min_nod = min_nod_0 # reset the MinNod value

        else:
            if link[row_index] != 0:
                link_pos, link_array, counter = self_link(link, row_index)
                for i in range(counter):
                    self_ref = link_array[i]
                    if self_ref <= stop:
                        c_indx_eq, erp_eq, min_nod, switch, chain_flag = eq_rod_assignment(erp_u, c_indx_u, c_indx_eq, erp_eq, min_nod, switch, self_ref, row_index, chain, min_nod_1)
                        if min_nod > row_index and min_nod != min_nod_0:
                            if row_index > stop + 1 and min_nod_1 != min_nod and chain[min_nod_1] == 0:
                                link[self_ref_1] = 0
                                # If the chain breaks, record where and which
                                # row to be reconnected
                                chain[min_nod_1] = link[min_nod_1]
                                self_ref_1 = self_ref
                                fill_in = min_nod # start the assign value to Link
                                while (link[fill_in] != 0) and link[fill_in] != fill_in:
                                    fill_in = link[fill_in]
                                if fill_in != self_ref_1:
                                    link[fill_in] = self_ref_1
                                else:
                                    min_nod = min_nod
                                chain[min_nod] = 0
                            elif row_index > stop + 1 and min_nod_1 != min_nod and chain_flag == 0:
                                link[self_ref_1] = 0
                        min_nod_1 = min_nod

                        min_nod = min_nod_0 # reset the MinNod value

            else:
                erp_eq[row_index + 1] = erp_eq[row_index]

            link[row_index] = 0 # zero the element in the Link list of current row

        # ready for next loop
        row_index = row_index + 1

    return erp_u, c_indx_u, erp_eq, c_indx_eq


def partial_num_lu(c_indx, c_indx_u, data, dim, erp, erp_u, stop, erp_eq, c_indx_eq, bound_bus):
    """Subroutine PartialNumLU performs partial numerical LU factorization to given full model bus addmittance matrix and calculates the equivalent branch reactance and the equivalent shunts (generated in the factorization process) added to the boundary buses.

    Inputs:
      CIndx  - N*1 array containning the column index of the rows, N is the number of non-zeros elements in the input data
      CIndxU - N*1 array containing the column index of off diagonal element in each row in the U matrix. The length N depends on the number of native plus filled non-zero elements in the off diagonal position in U matrix. CIndxU is unordered.
      Data   - N*1 array containing the data of matrix element in the original input file.
      dim    - scalar, dimension of the input matrix
      ERPU   - N_dim*1 array containing end of row pointer of all rows except the last row which doesn have any off diagonal element, N_dim is the dimension of the input matrix A in the original Ax = b problem
      ERP    - (N_node+1)*1 array containning the end of row pointer data
      stop   - scalar, equal to the number of external buses
      ERPEQ  - 1*n array, part of the pointers of the equivalent branches
      CIndxEQ- 1*n array, part of the pointers of the equivalent branches
      BoundBus- 1*n array, list of boundary buses

    Outputs:
      DataEQ  - 1*n array, includes reactance value of the equivalent branches
      DataShunt- 1*n array, includes equivalent bus shunts of the boundary buses
    """

    num_row = dim  # number of rows of given data matrix
    icpl = np.insert(erp_u[:-1] + 1, 0, 0)  # the initial column pointer equal to the last end of row pointer+1
    rx = 0  # Initiate the RX value;
    link = np.zeros(dim)  # initiate the Link array
    ex_acum = np.zeros(dim)  # Initiate the ExAcum;
    diag = np.zeros(num_row)
    data_eq = np.zeros(len(c_indx_eq))
    data_shunt = np.zeros(len(bound_bus))

    # Initialization based on ERPU, CindU;
    # Sort the CIndxU to make it Ordered CindUU->CindUO
    for i in range(1, num_row):
        row_col_ind = np.arange(erp_u[i - 1] + 1, erp_u[i] + 1)  # for every row the pointer of the column index
        c_indx_u[row_col_ind] = np.sort(c_indx_u[row_col_ind])  # sort the CIndx of every row in ascending order
    # begin Numerical Factorization
    row_index = 1  # Start at row 1
    while row_index <= num_row:
        # zero ExAcum using Link and CIndxUO;
        # This give the active element of current row
        # get the array from the self referential link
        if row_index > stop:
            ex_acum_eq = np.zeros(ex_acum.shape)
        if link[row_index - 1] != 0:
            link_pos, link_array, link_counter = self_link(link, row_index)
        else:
            link_counter = 0
        if link_counter != 0:
            if row_index < num_row:  # if this is the last row there will be nothing on the right of the diagonal in U only fills in row(numRow) of L
                ex_acum[np.concatenate([link_array, c_indx_u[np.arange(erp_u[row_index - 1] + 1, erp_u[
                    row_index] + 1)]])] = 0  # zero non-zero position from both native and fill
            else:
                ex_acum[link_array] = 0  # for last row, fill only
        else:
            ex_acum[c_indx_u[np.arange(erp_u[row_index - 1] + 1, erp_u[row_index] + 1)]] = 0
        # load corresponding values to ExAcum
        ex_acum[c_indx[np.arange(erp[row_index - 1] + 1, erp[row_index] + 1)]] = data[np.arange(erp[row_index - 1] + 1, erp[row_index] + 1)]  # Index in the original array is CIndx(ERP(RowIndex)+1:ERP(RowIndex+1))
        # step 2a
        rx = 0  # initiate RX
        # step 2b,c
        if link_counter != 0:
            link_array.sort()
            link_pos = link_pos[np.argsort(link_array)]
            link[link_pos] = 0
            for i in range(link_counter):
                rx = link_array[i]  # assign RX value to current fill generating row
                # step 2d
                if row_index > stop:
                    ex_acum_eq[c_indx_u[np.arange(erp_u[rx - 1] + 1, erp_u[rx] + 1)]] -= ex_acum[rx] * uro[
                        np.arange(erp_u[rx - 1] + 1, erp_u[rx] + 1)]
                ex_acum[c_indx_u[np.arange(erp_u[rx - 1] + 1, erp_u[rx] + 1)]] -= ex_acum[rx] * uro[
                    np.arange(erp_u[rx - 1] + 1, erp_u[rx] + 1)]
                # step 2ef
                lco[icpl[rx] - 1] = ex_acum[rx] * diag[rx - 1]
                icpl[rx] += 1
                # Link(LinkPos(i))=0;
                # step 2g
                if icpl[rx] <= erp_u[rx]:
                    self_ref = c_indx_u[icpl[rx] - 1]
                    while link[self_ref - 1] != 0:
                        self_ref = link[self_ref - 1]  # exhaust the link list and find a 0 position to store RX
                    link[self_ref - 1] = rx
        if row_index > stop:
            data_eq[np.arange(erp_eq[row_index - 1] + 1, erp_eq[row_index] + 1)] = 1 / ex_acum_eq[
                c_indx_eq[np.arange(erp_eq[row_index - 1] + 1, erp_eq[row_index] + 1)]]
            data_shunt[row_index - stop - 1] = ex_acum_eq[row_index - 1]
            # step 4
        if row_index <= stop:
            diag[row_index - 1] = 1 / ex_acum[row_index - 1]  # Invert the diagonal value
            # step 5
            if row_index < num_row:
                uro[np.arange(erp_u[row_index - 1] + 1, erp_u[row_index] + 1)] = ex_acum[c_indx_u[
                    np.arange(erp_u[row_index - 1] + 1, erp_u[row_index] + 1)]] * diag[
                                                                                     row_index - 1]  # Multiply active ExAcum by Diag(1) & store in URO
                # step 6
                self_ref = c_indx_u[icpl[row_index] - 1]
                while link[self_ref - 1] != 0:
                    self_ref = link[self_ref - 1]
                link[self_ref - 1] = row_index
        elif not np.any(link):
            break
        # Prepare for next loop
        row_index += 1  # Increment the RowIndex
    return data_eq, data_shunt


