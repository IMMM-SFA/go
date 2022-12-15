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


#### START HERE #####

def mpreduction(mpc, exbusorig, pf_flag):
    print('Reduction process start')
    print('Preprocess data')
    [mpc, exbusorig] = preprocessdata(mpc, exbusorig)
    dim = np.size(mpc.bus, 1)
    # check if dc terminals are external
    if isfield(mpc, 'dcline'):
        tf1 = np.in1d(mpc.dcline[:, 1], exbusorig)
        tf2 = np.in1d(mpc.dcline[:, 2], exbusorig)
        if (np.sum(tf1) + np.sum(tf2)) > 0:
            error('not able to eliminate HVDC line terminals')
    exbusorig = exbusorig.T
    if ~np.empty(exbusorig):
        print('\nConvert input data model')
        [nfrom, nto, branum, lineb, shuntb, bcirc, busnum, numb, selfb, mpc, exbus, newbusnum, oldbusnum] = initiation(
            mpc, exbusorig)  # ExBus with internal numbering
        # Create data structure
        print('\nCreating Y matrix of input full model')
        [cindx, erp, datab] = buildymat(nfrom, nto, branum, lineb, bcirc, busnum, numb, selfb)
        # Do Reduction
        print('\nDo first round reduction eliminating all external buses')
        [mpcreduced, bcircr, exbusr] = doreduction(datab, erp, cindx, exbus, numb, dim, bcirc, newbusnum, oldbusnum,
                                                   mpc)  # ExBusr with original numbering
        # Generate the second reduction with all retained buses and all generator
        # buses mpcreduced_gen
        # Create the ExBus_Gen to create the reduced model with all gens
        tf = np.in1d(exbus, mpc.gen[:, 1])
        exbusgen = exbus
        exbusgen[tf == 1] = []  # delete all external buses with generators
        tf = np.in1d(mpc.gen[:, 1], exbus)
        print('\n%d external generators are to be placed' % len(tf(tf == 1)))
        if ~np.empty(exbusgen):
            print('\nDo second round reduction eliminating all external non-generator buses')
            [mpcreduced_gen, bcirc_gen, exbusgen] = doreduction(datab, erp, cindx, exbusgen, numb, dim, bcirc,
                                                                newbusnum, oldbusnum, mpc)
        else:
            mpcreduced_gen = mpc
            mpcreduced_gen = mapbus(mpcreduced_gen, newbusnum, oldbusnum)
            bcirc_gen = bcirc
        # Move Generators
        print('\nPlacing External generators')
        [newgenbus, link] = moveexgen(mpcreduced_gen, exbusorig, exbusgen, bcirc_gen, 0)
        mpcreduced.gen[:, 1] = newgenbus  # move all external generators
        # Do Inverse PowerFlow
        print('\nRedistribute loads')
        mpc = mapbus(mpc, newbusnum, oldbusnum)
        [mpcreduced, bcircr] = loadredistribution(mpc, mpcreduced, bcircr, pf_flag)
    else:
        mpcreduced = mpc
        warning('No external buses, reduced model is same as full model')
    # Delete large reactance equivalent branches
    ind = np.where(np.abs(mpcreduced.branch[:, 4]) >= np.max(mpc.branch[:, 4]) * 10)
    mpcreduced.branch[ind, :] = []
    bcircr[ind] = []
    # Print Results
    print('\n**********Reduction Summary****************')
    print('\n%d buses in reduced model' % np.size(mpcreduced.bus, 1))
    print('\n%d branches in reduced model, including %d equivalent lines' % (
    np.size(mpcreduced.branch, 1), len(bcircr(bcircr == max(bcircr)))))
    print('\n%d generators in reduced model' % np.size(mpcreduced.gen, 1))
    if isfield(mpcreduced, 'dcline'):
        print('\n%d HVDC lines in reduced mode,' % np.size(mpcreduced.dcline, 1))
    print('\n**********Generator Placement Results**************')
    for i in range(np.size(link, 1)):
        if link[i, 2] - link[i, 1] != 0:
            print('\nExternal generator on bus %d is moved to %d' % (link[i, 1], link[i, 2]))
    print('\n')

    return mpcreduced, link, bcircr


def moveExGen(mpcreduced_gen, ExBus, ExBusGen, BCIRC, acflag):
  BranchRec = np.column_stack((mpcreduced_gen['branch'][:, [1,2]], BCIRC, mpcreduced_gen['branch'][:, [3,4]]))  # fnum,tnum,circuit,r,x
  if acflag == 0:
    BranchRec[:, 3] = 0  # for dc, ignore all resistance

  # Read the bus data
  BusNo = mpcreduced_gen['bus'][:, 1]

  # Convert original bus number to new bus number
  NewBusNo = np.arange(len(BusNo))
  BusRec = np.column_stack((NewBusNo,))
  BusRec = BusRec[BusRec[:, 0].argsort()]

  tf = np.isin(ExBus, ExBusGen)
  ExBus = ExBus[tf == 0]
  ExBus = np.interp(ExBus, BusNo, NewBusNo)
  tf = np.isin(BusRec[:, 0], ExBus)
  IntBus = BusRec[tf == 0, 0]
  BranchRec[:, :2] = np.interp(BranchRec[:, :2], BusNo, NewBusNo)
  BranchRec = BranchRec[BranchRec[:, [0,1]].argsort(axis=1)]

  Gen = mpcreduced_gen['gen']
  Gen[:, 0] = np.interp(Gen[:, 0], BusNo, NewBusNo)
  Gen = Gen[Gen[:, 0].argsort()]

  # clear num txt
  # Convert all parallel lines into single lines
  ignore, I = np.unique(BranchRec[:, [0,1]], axis=0, return_index=True)  # return the first unique rows in BranchRec
  idx = np.where(np.diff(I) != 1)[0]
  idx_del = []
  for k in idx:
    z = complex(BranchRec[I[k], 3], BranchRec[I[k], 4])  # complex value of impedances
    for kk in range(I[k]+1, I[k+1]):
      z1 = complex(BranchRec[kk, 3], BranchRec[kk, 4])
      z = 1 / (1/z + 1/z1)
    BranchRec[I[k], 3] = np.real(z)
    BranchRec[I[k], 4] = np.imag(z)
    idx_del = np.append(idx_del, range(I[k]+1, I[k+1]))
  BranchRec = np.delete(BranchRec, idx_del, 0)

  # Convert the external gen network into a radial network by Zmin
  GenNum = Gen[:, 0]
  tf = np.isin(GenNum, IntBus)
  GenNum[tf == 1] = []
  LinkedBus = np.zeros(BusNo.shape)
  LinkedBra = np.zeros(BusNo.shape)
  # Set up the levels
  Level = np.full(BusNo.shape, -1)
  Level[IntBus] = 0
  # Set up the distance
  Dist = np.full(BusNo.shape, np.inf)
  Dist[IntBus] = 0
  BranchZ = np.sqrt(np.square(BranchRec[:, 3]) + np.square(BranchRec[:, 4]))

  BusPrevLayer = IntBus
  BusTBD = GenNum

  for lev in range(1000):
    tf1 = np.isin(BranchRec[:, 0], BusPrevLayer)
    tf2 = np.isin(BranchRec[:, 1], BusTBD)
    ind = np.where(tf1 & tf2)[0]
    for k in ind:
        pi = BranchRec[k, 0]
        gi = BranchRec[k, 1]
        if Dist[gi] > BranchZ[k] + Dist[pi]:
            Dist[gi] = BranchZ[k] + Dist[pi]
            LinkedBus[gi] = pi
            LinkedBra[gi] = k
            Level[gi] = Level[pi] + 1

    tf1 = np.isin(BranchRec[:, 1], BusPrevLayer)
    tf2 = np.isin(BranchRec[:, 0], BusTBD)
    ind = np.where(tf1 & tf2)[0]
    for k in ind:
        pi = BranchRec[k, 1]
        gi = BranchRec[k, 0]
        if Dist[gi] > BranchZ[k] + Dist[pi]:
            Dist[gi] = BranchZ[k] + Dist[pi]
            LinkedBus[gi] = pi
            LinkedBra[gi] = k
            Level[gi] = Level[pi] + 1

    # Link to the internal bus with shortest path
    tf1 = np.isin(BranchRec[:, 0], BusTBD)
    tf2 = np.isin(BranchRec[:, 1], BusTBD)
    ind = np.where(tf1 & tf2)[0]

    for k in ind:
        pi = BranchRec[k, 0]
        gi = BranchRec[k, 1]

        if Dist[gi] > BranchZ[k] + Dist[pi]:
            Level[gi] = -1
        elif Dist[pi] > BranchZ[k] + Dist[gi]:
            Level[pi] = -1

  # LinkedBus=0 -> islanded buses       LinkedBus=-1
  LinkedBus[IntBus] = -1
  islanded_Bus = BusNo[np.where(LinkedBus == 0)]
  LinkedBus[np.where(LinkedBus == 0)] = 9999999

  for i in range(len(LinkedBus)):
    if LinkedBus[i] == -1:
      LinkedBus[i] = i

  BusNo = np.append(BusNo, 9999999)
  NewBusNo = np.append(NewBusNo, 9999999)
  islandflag = 1
  if len(LinkedBus[LinkedBus == 9999999]) == 0:
    islandflag = 0
    LinkedBus = np.append(LinkedBus, 9999999)

  LinkedBus = np.interp(LinkedBus, NewBusNo, BusNo)  # all the buses in the system and its correponding bus in the reduced system

  NewGenBus = np.interp(mpcreduced_gen['gen'][:, 0], BusNo, LinkedBus)
  if not islandflag:
    LinkedBus = np.delete(LinkedBus, LinkedBus == 9999999)
  Link = np.column_stack((mpcreduced_gen['bus'][:, 0], LinkedBus))

  return NewGenBus, Link


def MapBus(mpc, oldbusnum, newbusnum):
    # convert bus number
    mpc['bus'][:,0] = np.interp(oldbusnum, newbusnum, mpc['bus'][:,0])
    # convert branch terminal bus number
    mpc['branch'][:,0] = np.interp(oldbusnum, newbusnum, mpc['branch'][:,0])
    mpc['branch'][:,1] = np.interp(oldbusnum, newbusnum, mpc['branch'][:,1])
    # convert generator bus number
    mpc['gen'][:,0] = np.interp(oldbusnum, newbusnum, mpc['gen'][:,0])
#     if 'dcline' in mpc:
#         # convert hvdc line bus number
#         mpc['dcline'][:,0] = np.interp(oldbusnum, newbusnum, mpc['dcline'][:,0])
#         mpc['dcline'][:,1] = np.interp(oldbusnum, newbusnum, mpc['dcline'][:,1])
    return mpc


def MakeMPCr(ERPEQ, DataEQ, CIndxEQ, ShuntData, ERP, DataB, ExBus, PivInd, PivOrd, BCIRC, newbusnum, oldbusnum, mpcfull, BoundBus):
    ExLen = len(ExBus)
    # Create the reduced model case file
    mpcreduced = mpcfull
    branch = mpcreduced.branch
    bus = mpcreduced.bus
    int_flag = np.ones(len(branch),1)
    # delete all branches connect external buses
    # 1. eliminate all branches connecting external bus
    # check from bus
    for i in range(ExLen):
        tf = np.isin(branch[:,1],ExBus[i])
        int_flag = int_flag * ~tf
    # check to bus
    for i in range(ExLen):
        tf = np.isin(branch[:,2],ExBus[i])
        int_flag = int_flag * ~tf
    branch[int_flag==0] = [] # delete all marked branches
    BCIRC[int_flag==0] = []
    # delete all external buses
    for i in range(ExLen):
        bus[bus[:,1]==ExBus[i], :] = []
    # Generate data for equivalent branches
    FromInd = np.zeros(len(CIndxEQ))
    AddEqBranch = np.zeros((len(DataEQ), len(branch)))
    for i in range(ExLen+1, len(ERPEQ)-1):
        FromInd[ERPEQ[i]+1:ERPEQ[i+1]] = i
    for i in range(len(CIndxEQ)):
        AddEqBranch[i, [1,2,4]] = [PivInd[FromInd[i]], PivInd[CIndxEQ[i]], -DataEQ[i]]
    AddEqBranch[:,6] = 99999 # RATEA
    AddEqBranch[:,7] = 99999 # RATEB
    AddEqBranch[:,8] = 99999 # RATEC
    AddEqBranch[:,9] = 1 # tap
    AddEqBranch[:,10] = 0 # phase shift
    AddEqBranch[:,11] = 1 # status
    AddEqBranch[:,12] = -360 # min angle
    AddEqBranch[:,13] = 360
    # generate circuit number
    EqBCIRC = max(99, 10**(np.ceil(np.log10(max(BCIRC)-1)))-1)
    AddEqBCIRC = np.ones(len(AddEqBranch),1)*EqBCIRC
    branch = np.concatenate((branch, AddEqBranch))
    BCIRC = np.concatenate((BCIRC, AddEqBCIRC))
    mpcreduced.branch = branch
    # Calculate Bus Shunt
    BusShunt = np.zeros((len(mpcfull.bus)-ExLen,2))
    BusShunt[:,1] = np.arange(ExLen+1, len(mpcfull.bus))
    BusShunt[:,2] = DataB[ERP[BusShunt[:,1]]+1] # add original diagonal element in Y matrix of the bus in;
    BusShunt[:len(BoundBus),2] = BusShunt[:len(BoundBus),2] + ShuntData
    for i in range(len(branch)):
        m = PivOrd[branch[i,1]] - ExLen
        n = PivOrd[branch[i,2]] - ExLen
        BusShunt[m,2] = BusShunt[m,2] - 1/branch[i,4]
        BusShunt[n,2] = BusShunt[n,2] - 1/branch[i,4]
    BusShunt[:,1] = PivInd[BusShunt[:,1]]
    BusShunt[:,2] = BusShunt[:,2] * mpcfull.baseMVA
    # Plug the shunts value into the case file
    bus = np.sort(bus,axis=0)
    BusShunt = np.sort(BusShunt,axis=0)
    bus[:,6] = BusShunt[:,2]
    mpcreduced.bus = bus
    # covert all bus numbers back to original numbering
    mpcreduced.branch[:,5] = 0 # all branch shunts are converted to bus shunts
    mpcreduced.branch[:,1] = np.interp(mpcreduced.branch[:,1], newbusnum, oldbusnum)
    mpcreduced.branch[:,2] = np.interp(mpcreduced.branch[:,2], newbusnum, oldbusnum)
    mpcreduced.bus[:,1] = np.interp(mpcreduced.bus[:,1], newbusnum, oldbusnum)
    ExBus = np.interp(ExBus, newbusnum, oldbusnum)
    mpcreduced.gen[:,1] = np.interp(mpcreduced.gen[:,1], newbusnum, oldbusnum)

    return (mpcreduced, BCIRC, ExBus)


def LoadRedistribution(mpcfull, mpcreduced, BCIRCr, Pf_flag):
    if Pf_flag == 1:
        # OPT=mpoption('out.all',0);
        [resultfull,successfull]=rundcpf(mpcfull)
        if successfull == False:
            raise ValueError('unable to solve dc powerflow with original full model, load cannot be redistributed')
    else:
        resultfull = mpcfull
        successfull = 1

    # Read the full model bus data
    # [BusID, V_mag, V_angle
    OrigBusRec = resultfull.bus[:,[1,8,9]]
    OrigBusRec = np.sortrows(OrigBusRec,1) #reorder bus records

    # Read Bus Data
    BusRec = mpcreduced.bus
    BusRec = np.sortrows(BusRec,1)
    BusNo = BusRec[:,1]

    # Use original bus voltage
    ignore, ind = np.ismember(BusNo, OrigBusRec[:,1])
    BusRec[:,8] = OrigBusRec[ind,2] # Vm
    BusRec[:,9] = OrigBusRec[ind,3] # Vang

    # Read Branch DATA
    branchdata = mpcreduced.branch
    branchdata = branchdata[branchdata[:,11] == 1, :]
    BranchRec, braindex = np.sortrows(branchdata,[1,2])

    # Renumber the branch terminal buses
    Sbase = 100 #MVA
    NewBusNo = np.arange(len(BusNo))
    BusRec[:,1] = NewBusNo
    BranchRec[:,1] = np.interp(BusNo, NewBusNo, BranchRec[:,1])
    BranchRec[:,2] = np.interp(BusNo, NewBusNo, BranchRec[:,2])

    # read phase shifter information
    ind = np.where( np.abs(BranchRec[:,10]) )[0]
    flag = 0
    if len(ind) == 0:
        flag = 1
        phase_shifter = BranchRec[ind,:]

    # Form complex voltage vector
    Bus_V_Mag_PU = BusRec[:,8]
    Bus_V_Pha = BusRec[:,9]/180*np.pi

    # Form Y Matrix
    BB = np.zeros((len(BusNo),len(BusNo)))
    bb = BranchRec[:,4]
    BranchRec[BranchRec[:,9] == 0, 9] = 1
    bb = bb*(BranchRec[:,9]) # x/tap
    bb = 1/bb
    for i in range(len(BranchRec[:,4])):
        m = BranchRec[i,1]
        n = BranchRec[i,2]

        BB[m,m] = BB[m,m] + bb[i]
        BB[n,n] = BB[n,n] + bb[i]

        BB[m,n] = BB[m,n] - bb[i]
        BB[n,m] = BB[n,m] - bb[i]

    P_injected2 = BB.dot(Bus_V_Pha)*Sbase

    if flag == 1:
        # phase_shifter
        B_fix = np.zeros(len(BB[:,1]))
        for i in range(len(phase_shifter[:,1])):
            B_fix[ phase_shifter[i,1] ] = B_fix[ phase_shifter[i,1] ] - phase_shifter[i,10]*np.pi/180/phase_shifter[i,4]
            B_fix[ phase_shifter[i,2] ] = B_fix[ phase_shifter[i,2] ] + phase_shifter[i,10]*np.pi/180/phase_shifter[i,4]
        B_fix = B_fix*Sbase

    gen = mpcreduced.gen
    gen[:,2] = resultfull.gen[:,2] # use the full model solution
    gen[:,1] = np.interp(BusNo, NewBusNo, gen[:,1])
    Generation = np.zeros((mpcreduced.bus.shape[0],2))
    Generation[:,1] = NewBusNo
    for i in range(gen.shape[0]):
        Generation[gen[i,1],2] = Generation[gen[i,1],2] + gen[i,2]
    gen[:,1] = np.interp(NewBusNo, BusNo, gen[:,1])

    # fix the phase shifter
    if flag == 1:
        P_injected2 = P_injected2 + B_fix

    P_L_should = Generation[:,2] - P_injected2

    # dealing with HVDC lines
    if 'dcline' in mpcreduced:
        dcline = mpcfull.dcline
        HVDC_Line = [dcline[:,1],dcline[:,2],dcline[:,4],dcline[:,5]]
        HVDC_Line = np.sortrows(HVDC_Line,[1 2])
        HVDC_Line[:,1] = np.interp(BusNo, NewBusNo, HVDC_Line[:,1])
        HVDC_Line[:,2] = np.interp(BusNo, NewBusNo, HVDC_Line[:,2])
        # for HVDC lines if one bus of a line is isolated then the buses on the other end
        # of the line will be ignored in the inverse power flow program
        for i in range(len(HVDC_Line[:,1])):
            if (BusRec[HVDC_Line[i,1],2] != 4) and (BusRec[HVDC_Line[i,2],2] != 4):
                P_L_should[HVDC_Line[i,1]] = P_L_should[HVDC_Line[i,1]] - HVDC_Line[i,3]
                P_L_should[HVDC_Line[i,2]] = P_L_should[HVDC_Line[i,2]] + HVDC_Line[i,4] # YZ compensate HVDC line by adding/reducing the loads from the HVDC flows

    # Plug in the results
    mpcreduced.bus[:,3] = P_L_should
    mpcreduced.gen = gen

def initiation(mpc, ExBus):

    # sort the buses
    mpc['bus'] = np.sort(mpc['bus'], axis=0)
    oldbusnum = mpc['bus'][:,0]
    newbusnum = np.arange(1, len(mpc['bus'])+1)
    # change the branch terminal bus number
    mpc['branch'][:,0] = np.interp(oldbusnum, newbusnum, mpc['branch'][:,0])
    mpc['branch'][:,1] = np.interp(oldbusnum, newbusnum, mpc['branch'][:,1])
    mpc['gen'][:,0] = np.interp(oldbusnum, newbusnum, mpc['gen'][:,0])
    ExBus = np.interp(oldbusnum, newbusnum, ExBus)

    # bus data
    NUMB = newbusnum
    BusNum = len(mpc['bus'])
    SelfB = mpc['bus'][:,5] / mpc['baseMVA']
    # branch data
    BraNum = len(mpc['branch'])
    NFROM = mpc['branch'][:,0]
    NTO = mpc['branch'][:,1]
    LineB = 1 / mpc['branch'][:,3] # calculate the branch susceptance (b)
    ShuntB = mpc['branch'][:,4] / 2 # branch shunts
    BCIRC = generate_bcirc(mpc['branch'])
    # update SelfB
    for i in range(BraNum):
        SelfB[NFROM[i]] += LineB[i] + ShuntB[i]
        SelfB[NTO[i]] += LineB[i] + ShuntB[i]

    return NFROM, NTO, BraNum, LineB, ShuntB, BCIRC, BusNum, NUMB, SelfB, mpc, ExBus, newbusnum, oldbusnum


def generate_bcirc(branch):
    """GenerateBCIRC is used to detect parallel lines and generate the circuit number of every branch.

    Parameters
    ----------
    branch : matrix
        Includes branch data in MATPOWER case format.

    Returns
    -------
    BCIRC : n*1 vector
        Includes branch circuit number.

    Note
    ----
    For one branch, if its circuit number is greater than 1 then it is parallel to one of the branch whose circuit number is 1.
    """

    ft_num, m, n = np.unique(branch[:, [1, 2]], axis=0, return_index=True)
    n2, counts = np.unique(n, return_counts=True)
    bcirc = np.zeros(branch.shape[0], dtype=np.int)

    for i, count in enumerate(counts):
        ind = np.where(n == n2[i])[0]
        bcirc[ind] = np.arange(1, count + 1)

    return bcirc


def DoReduction(DataB,ERP,CIndx,ExBus,NUMB,dim,BCIRC,newbusnum,oldbusnum,mpc):

    # Define Boundary Buses
    BoundBus = DefBoundary(mpc,ExBus)

    # Do Pivot including Tinney One
    DataB,ERP,CIndx,PivOrd,PivInd = PivotData(DataB,ERP,CIndx,ExBus,NUMB,BoundBus)

    # Do LU factorization (Partial)
    ERPU,CIndxU,ERPEQ,CIndxEQ = PartialSymLU(CIndx,ERP,dim,len(ExBus),BoundBus)
    DataEQ,DataShunt = PartialNumLU (CIndx,CIndxU,DataB,dim,ERP,ERPU,len(ExBus),ERPEQ,CIndxEQ,BoundBus)

    # Create the reduced model in MATPOWER format
    mpcreduced,BCIRC,ExBus = MakeMPCr(ERPEQ,DataEQ,CIndxEQ,DataShunt,ERP,DataB,ExBus,PivInd,PivOrd,BCIRC,newbusnum,oldbusnum,mpc,BoundBus)

    return mpcreduced,BCIRC,ExBus


def DefBoundary(mpc, ExBus):
    # Subroutine DefBoundary indentify the boundary buses in the given model
    # mpc based on the list of external buses (ExBus).

    # INPUT DATA:
    #   mpc - struct, input system model in MATPOWER format
    #   ExBus - 1*n array, includes external bus indices

    # OUTPUT DATA:
    #   BoundBus - 1*n array, Boundary bus indices

    # Note:
    #   Boundary buses are the retained buses directly connected to external
    #   buses.

    BoundBus = np.zeros(mpc.bus.shape[0], dtype=int)
    ExFlag = BoundBus
    ExFlag[ExBus] = 1

    for i in range(mpc.branch.shape[0]):
        m = mpc.branch[i,0]
        n = mpc.branch[i,1]
        if ExFlag[m] + ExFlag[n] < 2: # exclude external branch
            if (ExFlag[m]*n + ExFlag[n]*m) != 0:
                BoundBus[ExFlag[m]*n + ExFlag[n]*m] = 1

    BoundBus = np.where(BoundBus == 1)[0]

    return BoundBus


def BuildYMat(NFROM, NTO, BraNum, LineB, BCIRC, BusNum, NUMB, SelfB):

  # Initialization
  ERP = np.arange(0, BusNum+1)

  # Read the branch one by one
  # First generate the ERP array
  for i in range(BraNum):
    if BCIRC[i] == 1:
      ERP[NFROM[i]+1:BusNum+1] += 1
      ERP[NTO[i]+1:BusNum+1] += 1

  # Second generate the CIndx and Data array
  DataB = np.zeros(ERP[BusNum+1])
  CIndx = np.zeros(ERP[BusNum+1])
  ICLP = ERP
  ICLP = ICLP + 1
  ICLP = np.delete(ICLP, BusNum+1)
  ICLP = np.insert(ICLP, 0, 0)
  CIndx[ICLP[1:BusNum+1]] = NUMB
  ICLP[1:BusNum+1] += 1

  for i in range(BraNum):
    DataB[ICLP[np.array([NFROM[i]+1, NTO[i]+1])]] -= LineB[i]
    if i < BraNum-1:
      if BCIRC[i+1] == 1:
        CIndx[ICLP[np.array([NFROM[i]+1, NTO[i]+1])]] = np.array([NTO[i], NFROM[i]])
        ICLP[np.array([NFROM[i]+1, NTO[i]+1])] += 1
    else:
      CIndx[ICLP[np.array([NFROM[i]+1, NTO[i]+1])]] = np.array([NTO[i], NFROM[i]])

  for i in range(BusNum):
    DataB[ERP[NUMB[i]]+1] += SelfB[i]

  return CIndx, ERP, DataB

