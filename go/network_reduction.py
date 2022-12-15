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


def move_ex_gen(mpcreduced_gen, ex_bus, ex_bus_gen, b_circ, ac_flag):
    """MoveExGen moves generators on external buses to internal buses based on
    shortest electrical distance strategy.

    Parameters
    ----------
    mpcreduced_gen : struct
        Reduced model with all non-generator external buses eliminated.
    ExBus : 1d array
        Includes all external bus indices.
    ExBusGen : 1d array
        Includes external generator bus indices.
    BCIRC : 1d array
        Includes branch circuit numbers in full model.
    acflag : scalar
        If 0, ignore all resistance; if 1, calculate electrical distance
        involving resistance.

    Returns
    -------
    NewGenBus : 1d array
        Includes new generator bus numbers after moving generators.
    Link : 2d array
        Generator mapping data of all generators. The first column is the
        original generator bus number and the second column is the new generator
        bus number after moving external generators.

    Notes
    -----
    The electrical distance between two buses are calcualted as sum of
    impedance in series connecting the two buses. If acflag = 0, the
    impedance is same as reactance.
    The shortest distance is found based on Dijkstra's algorithm.
    """

    branch_rec = np.column_stack((mpcreduced_gen['branch'][:, [1, 2]],
                                  b_circ,
                                  mpcreduced_gen['branch'][:, [3, 4]]))  # fnum,tnum,circuit,r,x
    if ac_flag == 0:
        branch_rec[:, 3] = 0  # for dc, ignore all resistance

    # Read the bus data
    bus_no = mpcreduced_gen['bus'][:, 1]

    # Convert original bus number to new bus number
    new_bus_no = np.arange(len(bus_no))
    bus_rec = np.column_stack((new_bus_no,))
    bus_rec = bus_rec[bus_rec[:, 0].argsort()]

    tf = np.isin(ex_bus, ex_bus_gen)
    ex_bus = ex_bus[tf == 0]
    ex_bus = np.interp(ex_bus, bus_no, new_bus_no)
    tf = np.isin(bus_rec[:, 0], ex_bus)
    int_bus = bus_rec[tf == 0, 0]
    branch_rec[:, :2] = np.interp(branch_rec[:, :2], bus_no, new_bus_no)
    branch_rec = branch_rec[branch_rec[:, [0, 1]].argsort(axis=1)]

    gen = mpcreduced_gen['gen']
    gen[:, 0] = np.interp(gen[:, 0], bus_no, new_bus_no)
    gen = gen[gen[:, 0].argsort()]

    # clear num txt
    # converter all parallel lines into single lines
    ignore, i = np.unique(branch_rec[:, [0, 1]], axis=0, return_index=True)  # return the first unique rows in BranchRec
    idx = np.where(np.diff(i) != 1)[0]
    idx_del = []
    for k in idx:
        z = np.complex(branch_rec[i[k], 2], branch_rec[i[k], 3])  # complex value of impedances
        for kk in range(i[k] + 1, i[k + 1] - 1):
            z1 = np.complex(branch_rec[kk, 2], branch_rec[kk, 3])
            z = 1 / (1 / z + 1 / z1)
        branch_rec[i[k], 2] = np.real(z)
        branch_rec[i[k], 3] = np.imag(z)
        idx_del = np.append(idx_del, range(i[k] + 1, i[k + 1] - 1))

    branch_rec = np.delete(branch_rec, idx_del, 0)

    # Convert the external gen network into a radial network by Zmin
    gen_num = gen[:, 0]
    tf = np.isin(gen_num, int_bus)
    gen_num[tf == 1] = []
    linked_bus = np.zeros(bus_no.shape)
    linked_bra = np.zeros(bus_no.shape)
    # Set up the levels
    level = np.full(bus_no.shape, -1)
    level[int_bus] = 0
    # Set up the distance
    dist = np.full(bus_no.shape, np.inf)
    dist[int_bus] = 0
    branch_z = np.sqrt(np.square(branch_rec[:, 3]) + np.square(branch_rec[:, 4]))

    bus_prev_layer = int_bus
    bus_tbd = gen_num

    for lev in range(1000):
        tf1 = np.isin(branch_rec[:, 0], bus_prev_layer)
        tf2 = np.isin(branch_rec[:, 1], bus_tbd)
        ind = np.where(tf1 & tf2)[0]
        for k in ind:
            pi = branch_rec[k, 0]
            gi = branch_rec[k, 1]
            if dist[gi] > branch_z[k] + dist[pi]:
                dist[gi] = branch_z[k] + dist[pi]
                linked_bus[gi] = pi
                linked_bra[gi] = k
                level[gi] = level[pi] + 1

        tf1 = np.isin(branch_rec[:, 1], bus_prev_layer)
        tf2 = np.isin(branch_rec[:, 0], bus_tbd)
        ind = np.where(tf1 & tf2)[0]
        for k in ind:
            pi = branch_rec[k, 1]
            gi = branch_rec[k, 0]
            if dist[gi] > branch_z[k] + dist[pi]:
                dist[gi] = branch_z[k] + dist[pi]
                linked_bus[gi] = pi
                linked_bra[gi] = k
                level[gi] = level[pi] + 1

        # Link to the internal bus with shortest path
        tf1 = np.isin(branch_rec[:, 0], bus_tbd)
        tf2 = np.isin(branch_rec[:, 1], bus_tbd)
        ind = np.where(tf1 & tf2)[0]

        for k in ind:
            pi = branch_rec[k, 0]
            gi = branch_rec[k, 1]

            if dist[gi] > branch_z[k] + dist[pi]:
                level[gi] = -1
            elif dist[pi] > branch_z[k] + dist[gi]:
                level[pi] = -1

    # LinkedBus=0 -> islanded buses       LinkedBus=-1
    linked_bus[int_bus] = -1
    islanded_bus = bus_no[np.where(linked_bus == 0)]
    linked_bus[np.where(linked_bus == 0)] = 9999999

    for i in range(len(linked_bus)):
        if linked_bus[i] == -1:
            linked_bus[i] = i

    bus_no = np.append(bus_no, 9999999)
    new_bus_no = np.append(new_bus_no, 9999999)
    island_flag = 1
    if len(linked_bus[linked_bus == 9999999]) == 0:
        island_flag = 0
        linked_bus = np.append(linked_bus, 9999999)

    linked_bus = np.interp(linked_bus,
                           new_bus_no,
                           bus_no)  # all the buses in the system and its correponding bus in the reduced system

    new_gen_bus = np.interp(mpcreduced_gen['gen'][:, 0], bus_no, linked_bus)

    if not island_flag:
        linked_bus = np.delete(linked_bus, linked_bus == 9999999)
    link = np.column_stack((mpcreduced_gen['bus'][:, 0], linked_bus))

    return new_gen_bus, link


def map_bus(mpc, old_busnum, new_busnum):
    """Subroutine MapBus converts bus indices from oldbusnum to newbusnum.
    The conversion will be done to fields including buses, branches, and generators.

    Parameters
    ----------
    mpc : struct
        Input model in MATPOWER format
    oldbusnum : array
        1*n array of the old bus indices which will be converted "from"
    newbusnum : array
        1*n array of the new bus indices which will be converted "to"

    Returns
    -------
    mpc : struct
        Output model in MATPOWER format with converted bus indices.
    """
    # Convert bus number
    mpc['bus'][:, 0] = np.interp(old_busnum, new_busnum, mpc['bus'][:, 0])

    # Convert branch terminal bus number
    mpc['branch'][:, 0] = np.interp(old_busnum, new_busnum, mpc['branch'][:, 0])
    mpc['branch'][:, 1] = np.interp(old_busnum, new_busnum, mpc['branch'][:, 1])

    # Convert generator bus number
    mpc['gen'][:, 0] = np.interp(old_busnum, new_busnum, mpc['gen'][:, 0])

    # if 'dcline' in mpc:
    #     # Convert hvdc line bus number
    #     mpc['dcline'][:, 0] = np.interp(old_busnum, new_busnum, mpc['dcline'][:, 0])
    #     mpc['dcline'][:, 1] = np.interp(old_busnum, new_busnum, mpc['dcline'][:, 1])

    return mpc


def make_mpcr(erpeq, dataeq, cindxeq, shuntdata, erp, datab, exbus, pivind, pivord, bcirc,
              newbusnum, oldbusnum, mpcfull, boundbus):
    """Subroutine MakeMPCr generates the reduced model in MATPOWER case format
    without generator placement and load redistribution.

    Parameters
    ----------
    ERPEQ : 1*n array
        End of row pointers of the equivalent lines.
    DataEQ : 1*n array
        Value of equivalent line reactance.
    CIndxEQ : 1*n array
        Column indices of equivalent lines.
    ShuntData : 1*n array
        Bus shunts data of all buses in the reduced model.
    ERP : 1*n array
        End of row pointer of the original full model bus admittance matrix.
    DataB : 1*n array
        Value of all non-zeros in the original full model bus admittance matrix.
    ExBus : 1*n array
        Indices of external buses.
    PivOrd : 1*n array
        Bus indices after pivotting.
    PivInd : 1*n array
        Bus ordering after pivotting.
    BCIRC : 1*n array
        Branch circuit number of the full model.
    newbusnum : 1*n array
        Internal bus indices.
    oldbusnum : 1*n array
        Original bus indices.
    mpcfull : struct
        The original full model.
    BoundBus : 1*n array
        Indices of boundary buses.

    Returns
    -------
    mpcreduced : struct
        The reduced model.
    BCIRC : 1*n array
        The branch circuit number of the reduced model.
    ExBus : 1*n array
        The external bus indices.

    Note
    ----
    The output data of this subroutine will be converted to original indices.
    """

    exlen = len(exbus)

    # Create the reduced model case file
    mpcreduced = mpcfull
    branch = mpcreduced.branch
    bus = mpcreduced.bus
    int_flag = np.ones(len(branch), 1)

    # delete all branches connect external buses
    # 1. eliminate all branches connecting external bus
    # check from bus
    for i in range(exlen):
        tf = np.isin(branch[:, 1], exbus[i])
        int_flag = int_flag * ~tf

    # check to bus
    for i in range(exlen):
        tf = np.isin(branch[:, 2], exbus[i])
        int_flag = int_flag * ~tf
    branch[int_flag == 0] = []  # delete all marked branches
    bcirc[int_flag == 0] = []

    # delete all external buses
    for i in range(exlen):
        bus[bus[:, 1] == exbus[i], :] = []

    # Generate data for equivalent branches
    from_ind = np.zeros(len(cindxeq))
    add_eq_branch = np.zeros((len(dataeq), len(branch)))
    for i in range(exlen + 1, len(erpeq) - 1):
        from_ind[erpeq[i] + 1:erpeq[i + 1]] = i
    for i in range(len(cindxeq)):
        add_eq_branch[i, [1, 2, 4]] = [pivind[from_ind[i]], pivind[cindxeq[i]], -dataeq[i]]
    add_eq_branch[:, 6] = 99999  # RATEA
    add_eq_branch[:, 7] = 99999  # RATEB
    add_eq_branch[:, 8] = 99999  # RATEC
    add_eq_branch[:, 9] = 1  # tap
    add_eq_branch[:, 10] = 0  # phase shift
    add_eq_branch[:, 11] = 1  # status
    add_eq_branch[:, 12] = -360  # min angle
    add_eq_branch[:, 13] = 360

    # generate circuit number
    eq_bcirc = max(99, 10**(np.ceil(np.log10(max(bcirc) - 1))) - 1)
    add_eq_bcirc = np.ones(len(add_eq_branch), 1)*eq_bcirc
    branch = np.concatenate((branch, add_eq_branch))
    bcirc = np.concatenate((bcirc, add_eq_bcirc))
    mpcreduced.branch = branch

    # Calculate Bus Shunt
    bus_shunt = np.zeros((len(mpcfull.bus) - exlen, 2))
    bus_shunt[:, 1] = np.arange(exlen + 1, len(mpcfull.bus))
    bus_shunt[:, 2] = datab[erp[bus_shunt[:, 1]] + 1]  # add original diagonal element in Y matrix of the bus in;
    bus_shunt[:len(boundbus), 2] = bus_shunt[:len(boundbus), 2] + shuntdata
    for i in range(len(branch)):
        m = pivord[branch[i, 1]] - exlen
        n = pivord[branch[i, 2]] - exlen
        bus_shunt[m, 2] = bus_shunt[m, 2] - 1/branch[i, 4]
        bus_shunt[n, 2] = bus_shunt[n, 2] - 1/branch[i, 4]
    bus_shunt[:, 1] = pivind[bus_shunt[:, 1]]
    bus_shunt[:, 2] = bus_shunt[:, 2] * mpcfull.baseMVA

    # Plug the shunts value into the case file
    bus = np.sort(bus, axis=0)
    bus_shunt = np.sort(bus_shunt, axis=0)
    bus[:, 6] = bus_shunt[:, 2]
    mpcreduced.bus = bus

    # covert all bus numbers back to original numbering
    mpcreduced.branch[:, 5] = 0  # all branch shunts are converted to bus shunts
    mpcreduced.branch[:, 1] = np.interp(mpcreduced.branch[:, 1], newbusnum, oldbusnum)
    mpcreduced.branch[:, 2] = np.interp(mpcreduced.branch[:, 2], newbusnum, oldbusnum)
    mpcreduced.bus[:, 1] = np.interp(mpcreduced.bus[:, 1], newbusnum, oldbusnum)
    exbus = np.interp(exbus, newbusnum, oldbusnum)
    mpcreduced.gen[:, 1] = np.interp(mpcreduced.gen[:, 1], newbusnum, oldbusnum)

    return mpcreduced, bcirc, exbus

def load_redistribution(mpcfull, mpcreduced, bcircr, pf_flag):
    """Subroutine LoadRedistribution moves loads in reduced model in order to
    make the dcpf solution on reduced model identical to the full model with
    external generator placed in subroutine MoveExGen.

    Parameters
    ----------
    mpcfull : struct
        Full model data in MATPOWER case format
    mpcreduced : struct
        Reduced model data with all external buses eliminated in MATPOWER case format
    BCIRCr : 1*n array
        Includes circuit number of branches in reduced model
    Pf_flag : scalar
        Indicates if dc power flow need to be solved before load redistribution

    Returns
    -------
    mpcreduced : struct
        Reduced model data with load redistributed
    BCIRCr : 1*n array
        Includes reordered branch circuit number in reduced model

    Note
    ----
    The subroutine will first run a dc power flow on the full model. If the
    dc power flow can not be solved on the full model, the subroutine will
    be terminated and an error will be returned.
    """

    if pf_flag == 1:
        # OPT=mpoption('out.all',0);
        resultfull, successfull = rundcpf(mpcfull)
        if successfull == 0:
            raise Exception('Unable to solve dc powerflow with original full model, load cannot be redistributed')
    else:
        resultfull = mpcfull
        successfull = 1

    # Read the full model bus data
    # [BusID, V_mag, V_angle
    orig_bus_rec = resultfull['bus'][:, [1, 8, 9]]
    orig_bus_rec = np.sort(orig_bus_rec, axis=0) # reorder bus records

    # Read Bus Data
    bus_rec = mpcreduced['bus']
    bus_rec = np.sort(bus_rec, axis=0)
    bus_no = bus_rec[:, 1]

    # Use original bus voltage
    ind = np.in1d(bus_no, orig_bus_rec[:, 1])
    bus_rec[:, 8] = orig_bus_rec[ind, 2] # Vm
    bus_rec[:, 9] = orig_bus_rec[ind, 3] # Vang

    # Read Branch DATA
    branchdata = mpcreduced['branch']
    branchdata = branchdata[np.where(branchdata[:, 11]==1)[0], :]
    branchdata, braindex = np.sort(branchdata, axis=0, kind='mergesort')
    branch_rec = branchdata

    # Renumber the branch terminal buses
    sbase = 100 # MVA
    new_bus_no = np.arange(1, len(bus_no)+1)
    bus_rec[:, 1] = new_bus_no
    branch_rec[:, 1] = np.interp(bus_no, new_bus_no, branch_rec[:, 1])
    branch_rec[:, 2] = np.interp(bus_no, new_bus_no, branch_rec[:, 2])

    # Read phase shifter information
    ind = np.where(abs(branch_rec[:, 10]))[0]
    flag = 0
    if len(ind) == 0:
        flag = 1
        phase_shifter = branch_rec[ind, :]

    # Form complex voltage vector
    bus_v_mag_pu = bus_rec[:, 8]
    bus_v_pha = bus_rec[:, 9]/180*np.pi

    # Form Y Matrix
    bb = np.zeros((len(bus_no), len(bus_no)))
    bb = branch_rec[:, 4]
    branch_rec[np.where(branch_rec[:, 9]==0)[0], 9] = 1
    bb = bb*(branch_rec[:, 9]) # x/tap
    bb = 1/bb
    for i in range(len(branch_rec[:, 4])):
        m = branch_rec[i, 1]
        n = branch_rec[i, 2]

        bb[m, m] += bb[i]
        bb[n, n] += bb[i]

        bb[m, n] -= bb[i]
        bb[n, m] -= bb[i]

    p_injected2 = bb*bus_v_pha*sbase

    if flag == 1:
        # phase_shifter
        b_fix = np.zeros((len(bb[:, 1]), 1))
        for i in range(len(phase_shifter[:, 1])):
            b_fix[phase_shifter[i, 1]] -= phase_shifter[i, 10]*np.pi/180/phase_shifter[i, 4]
            b_fix[phase_shifter[i, 2]] += phase_shifter[i, 10]*np.pi/180/phase_shifter[i, 4]
        b_fix = b_fix*sbase

    gen = mpcreduced['gen']
    gen[:, 2] = resultfull['gen'][:, 2] # use the full model solution
    gen[:, 1] = np.interp(bus_no, new_bus_no, gen[:, 1])
    generation = np.zeros((mpcreduced['bus'].shape[1], 2))
    generation[:, 1] = new_bus_no
    for i in range(gen.shape[1]):
        generation[gen[i, 1], 2] += gen[i, 2]
    gen[:, 1] = np.interp(new_bus_no, bus_no, gen[:, 1])

    # Fix the phase shifter
    if flag == 1:
        p_injected2 += b_fix

    p_l_should = generation[:, 2]-p_injected2

    # dealing with HVDC lines
    if "dcline" in mpcreduced.keys():
        dcline = mpcfull['dcline']
        hvdc_line = [dcline[:, 1], dcline[:, 2], dcline[:, 4], dcline[:, 5]]
        hvdc_line = np.sort(hvdc_line, axis=0)
        hvdc_line[:, 1] = np.interp(bus_no, new_bus_no, hvdc_line[:, 1])
        hvdc_line[:, 2] = np.interp(bus_no, new_bus_no, hvdc_line[:, 2])
        # for HVDC lines if one bus of a line is isolated then the buses on the other end
        # of the line will be ignored in the inverse power flow program
        for i in range(hvdc_line.shape[1]):
            if (bus_rec[hvdc_line[i, 1], 2] != 4) and (bus_rec[hvdc_line[i, 2], 2] != 4):
                p_l_should[hvdc_line[i, 1]] -= hvdc_line[i, 3]
                p_l_should[hvdc_line[i, 2]] += hvdc_line[i, 4] # YZ compensate HVDC line by adding/reducing the loads from the HVDC flows

    # Plug in the results
    mpcreduced['bus'][:, 3] = p_l_should
    mpcreduced['gen'] = gen

    return mpcreduced, bcircr


def initiation(mpc, ex_bus):
    """Converts full model data in MATPOWER case format to generate the full model admittance matrix.

    Parameters
    ----------
    mpc : struct
        Full model data in MATPOWER case format.
    ExBus : n*1 vector
        Includes indices of external buses.

    Returns
    -------
    NFROM : 1*n array
        Includes indices of all from end buses.
    NTO : 1*n array
        Includes indices of all to end buses.
    BraNum : 1*n array
        Includes indices of all branches.
    LineB : 1*n array
        Includes line admittance of branches.
    ShuntB : 1*n array
        Includes line shunt admittance of branches.
    mpc : struct
        Full model in MATPOWER case format.
    ExBus : n*1 vector
        Includes external bus indices.
    newbusnum : 1*n array
        Indices of buses in internal bus numbering.
    oldbusnum : 1*n array
        Indices of buses in original bus numbering.

    Notes
    -----
    All output bus indices are in internal bus numbering.
    """

    # Sort the buses
    mpc['bus'] = np.sort(mpc['bus'], axis=0)
    oldbusnum = mpc['bus'][:, 0]
    newbusnum = np.arange(1, len(mpc['bus']) + 1)

    # Change the branch terminal bus number
    mpc['branch'][:, 0] = np.interp(oldbusnum, newbusnum, mpc['branch'][:, 0])
    mpc['branch'][:, 1] = np.interp(oldbusnum, newbusnum, mpc['branch'][:, 1])
    mpc['gen'][:, 0] = np.interp(oldbusnum, newbusnum, mpc['gen'][:, 0])
    ex_bus = np.interp(oldbusnum, newbusnum, ex_bus)

    # Bus data
    numb = newbusnum
    bus_num = len(mpc['bus'])
    self_b = mpc['bus'][:, 5] / mpc['baseMVA']

    # Branch data
    bra_num = len(mpc['branch'])
    n_from = mpc['branch'][:, 0]
    nto = mpc['branch'][:, 1]
    line_b = 1 / mpc['branch'][:, 3]  # Calculate the branch susceptance (b)
    shubt_b = mpc['branch'][:, 4] / 2  # Branch shunts
    bcirc = generate_bcirc(mpc['branch'])

    # Update SelfB
    for i in range(BraNum):
        self_b[n_from[i]] += line_b[i] + shubt_b[i]
        self_b[nto[i]] += line_b[i] + shubt_b[i]

    return n_from, nto, bra_num, line_b, shubt_b, bcirc, bus_num, numb, self_b, mpc, ex_bus, newbusnum, oldbusnum

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


def do_reduction(data_b, erp, c_indx, ex_bus, numb, dim, bcirc, new_bus_num, old_bus_num, mpc):
    """Create the reduced network based on input network data.

    Parameters
    ----------
    DataB : 1*n array
        Admittance value in input admittance matrix
    ERP : 1*n array
        End of row pointer of the input admittance matrix
    CIndx : 1*n array
        Column indices of every row of the input admittance matrix
    ExBus : 1*n array
        External bus indices
    NUMB : 1*n array
        Bus indices
    dim : scalar
        Dimension of the input admittance matrix (should be square)
    BCIRC : 1*n array
        Branch circuit number
    newbusnum : 1*n array
        Internal bus indices
    oldbusnum : 1*n array
        Original bus indices

    Returns
    -------
    mpcreduced : struct
        Reduced model, without external generator placement and load redistribution
    BCIRC : 1*n array
        Updated branch circuit number
    ExBus : 1*n array
        Updated external bus indices

    Note
    ----
    The reduced model generated by this subroutine doesn't involve
    external generator placement and load redistribution. It's only good for
    analyze the reduced network (toplogy+reactance).
    """

    # Define Boundary Buses
    bound_bus = def_boundary(mpc, ex_bus)

    # Do Pivot including Tinney One
    data_b, erp, c_indx, piv_ord, piv_ind = pivot_data(data_b, erp, c_indx, ex_bus, numb, bound_bus)

    # Do LU factorization (Partial)
    erpu, c_indxu, erpeq, c_indxeq = partial_sym_lu(c_indx, erp, dim, len(ex_bus), bound_bus)
    data_eq, data_shunt = partial_num_lu(c_indx, c_indxu, data_b, dim, erp, erpu, len(ex_bus), erpeq, c_indxeq, bound_bus)

    # Create the reduced model in MATPOWER format
    mpcreduced, bcirc, ex_bus = make_m_p_cr(erpeq, data_eq, c_indxeq, data_shunt, erp, data_b, ex_bus, piv_ind, piv_ord, bcirc, new_bus_num, old_bus_num, mpc, bound_bus)

    return mpcreduced, bcirc, ex_bus


def def_boundary(mpc, ex_bus):
    """Identify the boundary buses in the given model, mpc, based on the list of external buses (ExBus).

    Parameters
    ----------
    mpc : struct
        Input system model in MATPOWER format
    ExBus : array
        1*n array, includes external bus indices

    Returns
    -------
    BoundBus : array
        1*n array, Boundary bus indices

    Note
    ----
    Boundary buses are the retained buses directly connected to external buses.
    """

    bound_bus = np.zeros(mpc.bus.shape[0], dtype=int)
    ex_flag = bound_bus
    ex_flag[ex_bus] = 1

    for i in range(mpc.branch.shape[0]):
        m = mpc.branch[i, 0]
        n = mpc.branch[i, 1]
        if ex_flag[m] + ex_flag[n] < 2:  # exclude external branch
            if (ex_flag[m] * n + ex_flag[n] * m) != 0:
                bound_bus[ex_flag[m] * n + ex_flag[n] * m] = 1

    bound_bus = np.where(bound_bus == 1)[0]

    return bound_bus


def build_y_mat(n_from, n_to, bra_num, line_b, b_circ, bus_num, numb, self_b):
    """Subroutine BuildYMat constructs an admittance matrix and stores it in a
    compact storage format to facilitate the use of sparse techniques.

    Parameters
    ----------
        NFROM: 1xn array containing the bus indices of the from end buses of
            every branch
        NTO: 1xn array containing the bus indices of the to end buses of
            every branch
        BraNum: scalar containing the number of branches
        NUMB: 1xn array containing the bus indices
        SelfB: 1xn array containing the total B shunt on every bus (B shunt on
            bus and half the branch B shunt)

    Returns
    -------
        CIndx: 1xn array containing the column indices of every row in the
            admittance matrix
        ERP: 1xn array containing the end of row pointers of the admittance
            matrix
        DataB: 1xn array containing the admittance values in the admittance
            matrix
    """

    # Initialization
    erp = np.arange(0, bus_num+1)

    # Read the branch one by one
    # First generate the ERP array
    for i in range(bra_num):
    if b_circ[i] == 1:
      erp[n_from[i]+1:bus_num+1] += 1
      erp[n_to[i]+1:bus_num+1] += 1

    # Second generate the CIndx and Data array
    data_b = np.zeros(erp[bus_num+1])
    c_indx = np.zeros(erp[bus_num+1])
    iclp = erp
    iclp = iclp + 1
    iclp = np.delete(iclp, bus_num+1)
    iclp = np.insert(iclp, 0, 0)
    c_indx[iclp[1:bus_num+1]] = numb
    iclp[1:bus_num+1] += 1

    for i in range(bra_num):
    data_b[iclp[np.array([n_from[i]+1, n_to[i]+1])]] -= line_b[i]
    if i < bra_num-1:
      if b_circ[i+1] == 1:
        c_indx[iclp[np.array([n_from[i]+1, n_to[i]+1])]] = np.array([n_to[i], n_from[i]])
        iclp[np.array([n_from[i]+1, n_to[i]+1])] += 1
    else:
      c_indx[iclp[np.array([n_from[i]+1, n_to[i]+1])]] = np.array([n_to[i], n_from[i]])

    for i in range(bus_num):
    data_b[erp[numb[i]]+1] += self_b[i]

    return c_indx, erp, data_b
