import numpy as np
import itertools
import copy
import pax
import jax
import jax.numpy as jnp

class Board(pax.Module):
    init_board: jnp.ndarray
    mapping: dict
    from_pieces: dict

    def __init__(self):
        init_board = np.zeros([8,8], dtype=np.int32).astype(str)
        self.mapping = {"r": 1, "n": 2, "b": 3, "q": 4, "k": 5, "p": 6, "R": 7, "N": 8, "B": 9, "Q": 10, "K": 11, "P": 12, " ": 0}
        self.from_pieces = {v:k for k,v in self.mapping.items()}
        init_board[0,0] = "r"
        init_board[0,1] = "n"
        init_board[0,2] = "b"
        init_board[0,3] = "q"
        init_board[0,4] = "k"
        init_board[0,5] = "b"
        init_board[0,6] = "n"
        init_board[0,7] = "r"
        init_board[1,0:8] = "p"
        init_board[7,0] = "R"
        init_board[7,1] = "N"
        init_board[7,2] = "B"
        init_board[7,3] = "Q"
        init_board[7,4] = "K"
        init_board[7,5] = "B"
        init_board[7,6] = "N"
        init_board[7,7] = "R"
        init_board[6,0:8] = "P"
        init_board[init_board == "0"] = " "

        self.board = jnp.zeros((8,8), dtype=jnp.int32)
        for i in range(8):
            for j in range(8):
                self.board = self.board.at[i, j].set(self.mapping[init_board[i,j]])
        self.move_count = 0
        self.no_progress_count = 0
        self.repetitions_w = 0
        self.repetitions_b = 0
        self.move_history = None
        self.en_passant = -999; self.en_passant_move = 0 # returns j index of last en_passant pawn
        self.r1_move_count = 0 # black's queenside rook
        self.r2_move_count = 0 # black's kingside rook
        self.k_move_count = 0
        self.R1_move_count = 0 # white's queenside rook
        self.R2_move_count = 0 # white's kingside rook
        self.K_move_count = 0
        self.current_board = self.board
        self.en_passant_move_copy = None
        self.copy_board = None; self.en_passant_copy = None; self.r1_move_count_copy = None; self.r2_move_count_copy = None; 
        self.k_move_count_copy = None; self.R1_move_count_copy = None; self.R2_move_count_copy = None; self.K_move_count_copy = None
        self.player = 0 


    def transform_board(self):
        board = self.current_board

        board_res = np.zeros((8,8), dtype=np.int32).astype(str)
        m = self.from_pieces

        for i in range(8):
            for j in range(8):
                board_res[i, j] = m(board[i,j])

        return board_res

    def inverse_board(self, cb):

        res = jnp.zeros((8, 8), dtype=jnp.int32)

        for i in range(8):
            for j in range(8):
                res[i, j] = self.mapping[cb[i, j]]

        return res
    
    def move_rules_P(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.transform_board()
        ## to calculate allowed moves for king
        threats = []
        if 0<=i-1<=7 and 0<=j+1<=7:
            threats.append((i-1,j+1))
        if 0<=i-1<=7 and 0<=j-1<=7:
            threats.append((i-1,j-1))
        #at initial position
        if i==6:
            if board_state[i-1,j] ==" ":
                next_positions.append((i-1,j))
                if board_state[i-2,j]==" ":
                    next_positions.append((i-2,j))
        # en passant capture
        elif i==3 and self.en_passant!=-999:
            if j-1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i-1,j-1))
            elif j+1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i-1,j+1))
        if i in [1,2,3,4,5] and board_state[i-1,j]==" ":
            next_positions.append((i-1,j))          
        if j==0 and board_state[i-1,j+1] in ["r", "n", "b", "q", "k", "p"]:
            next_positions.append((i-1,j+1))
        elif j==7 and board_state[i-1,j-1] in ["r", "n", "b", "q", "k", "p"]:
            next_positions.append((i-1,j-1))
        elif j in [1,2,3,4,5,6]:
            if board_state[i-1,j+1] in ["r", "n", "b", "q", "k", "p"]:
                next_positions.append((i-1,j+1))
            if board_state[i-1,j-1] in ["r", "n", "b", "q", "k", "p"]:
                next_positions.append((i-1,j-1))
        return next_positions, threats    

    def move_rules_p(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.transform_board()
        ## to calculate allowed moves for king
        threats = []
        if 0<=i+1<=7 and 0<=j+1<=7:
            threats.append((i+1,j+1))
        if 0<=i+1<=7 and 0<=j-1<=7:
            threats.append((i+1,j-1))
        #at initial position
        if i==1:
            if board_state[i+1,j]==" ":
                next_positions.append((i+1,j))
                if board_state[i+2,j]==" ":
                    next_positions.append((i+2,j))
        # en passant capture
        elif i==4 and self.en_passant!=-999:
            if j-1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i+1,j-1))
            elif j+1==self.en_passant and abs(self.en_passant_move-self.move_count) == 1:
                next_positions.append((i+1,j+1))
        if i in [2,3,4,5,6] and board_state[i+1,j]==" ":
            next_positions.append((i+1,j))          
        if j==0 and board_state[i+1,j+1] in ["R", "N", "B", "Q", "K", "P"]:
            next_positions.append((i+1,j+1))
        elif j==7 and board_state[i+1,j-1] in ["R", "N", "B", "Q", "K", "P"]:
            next_positions.append((i+1,j-1))
        elif j in [1,2,3,4,5,6]:
            if board_state[i+1,j+1] in ["R", "N", "B", "Q", "K", "P"]:
                next_positions.append((i+1,j+1))
            if board_state[i+1,j-1] in ["R", "N", "B", "Q", "K", "P"]:
                next_positions.append((i+1,j-1))
        return next_positions, threats

    def move_rules_r(self,current_position):
        i, j = current_position
        board_state = self.transform_board()
        next_positions = []; a=i
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        return next_positions


    def move_rules_R(self,current_position):
        i, j = current_position
        board_state = self.transform_board()
        next_positions = []; a=i
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        return next_positions

    def move_rules_n(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.transform_board()
        for a,b in [(i+2,j-1),(i+2,j+1),(i+1,j-2),(i-1,j-2),(i-2,j+1),(i-2,j-1),(i-1,j+2),(i+1,j+2)]:
            if 0<=a<=7 and 0<=b<=7:
                if board_state[a,b] in ["R", "N", "B", "Q", "K", "P", " "]:
                    next_positions.append((a,b))
        return next_positions

    def move_rules_N(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.transform_board()
        for a,b in [(i+2,j-1),(i+2,j+1),(i+1,j-2),(i-1,j-2),(i-2,j+1),(i-2,j-1),(i-1,j+2),(i+1,j+2)]:
            if 0<=a<=7 and 0<=b<=7:
                if board_state[a,b] in ["r", "n", "b", "q", "k", "p", " "]:
                    next_positions.append((a,b))
        return next_positions

    def move_rules_b(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.transform_board()
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions

    def move_rules_B(self,current_position):
        i, j = current_position
        next_positions = []
        board_state = self.transform_board()
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return 
        
    def move_rules_q(self,current_position):
        i, j = current_position
        board_state = self.transform_board()
        next_positions = [];a=i
        #bishop moves
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        #rook moves
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["R", "N", "B", "Q", "K", "P"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions

    def move_rules_Q(self,current_position):
        i, j = current_position
        board_state = self.transform_board()
        next_positions = [];a=i
        #bishop moves
        while a!=0:
            if board_state[a-1,j]!=" ":
                if board_state[a-1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,j))
                break
            next_positions.append((a-1,j))
            a-=1
        a=i
        while a!=7:
            if board_state[a+1,j]!=" ":
                if board_state[a+1,j] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,j))
                break
            next_positions.append((a+1,j))
            a+=1
        a=j
        while a!=7:
            if board_state[i,a+1]!=" ":
                if board_state[i,a+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a+1))
                break
            next_positions.append((i,a+1))
            a+=1
        a=j
        while a!=0:
            if board_state[i,a-1]!=" ":
                if board_state[i,a-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((i,a-1))
                break
            next_positions.append((i,a-1))
            a-=1
        #rook moves
        a=i;b=j
        while a!=0 and b!=0:
            if board_state[a-1,b-1]!=" ":
                if board_state[a-1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b-1))
                break
            next_positions.append((a-1,b-1))
            a-=1;b-=1
        a=i;b=j
        while a!=7 and b!=7:
            if board_state[a+1,b+1]!=" ":
                if board_state[a+1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b+1))
                break
            next_positions.append((a+1,b+1))
            a+=1;b+=1
        a=i;b=j
        while a!=0 and b!=7:
            if board_state[a-1,b+1]!=" ":
                if board_state[a-1,b+1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a-1,b+1))
                break
            next_positions.append((a-1,b+1))
            a-=1;b+=1
        a=i;b=j
        while a!=7 and b!=0:
            if board_state[a+1,b-1]!=" ":
                if board_state[a+1,b-1] in ["r", "n", "b", "q", "k", "p"]:
                    next_positions.append((a+1,b-1))
                break
            next_positions.append((a+1,b-1))
            a+=1;b-=1
        return next_positions
    
    def possible_W_moves(self, threats=False):
        board_state = self.transform_board()
        rooks = {}; knights = {}; bishops = {}; queens = {}; pawns = {};
        i,j = np.where(board_state=="R")
        for rook in zip(i,j):
            rooks[tuple(rook)] = self.move_rules_R(rook)
        i,j = np.where(board_state=="N")
        for knight in zip(i,j):
            knights[tuple(knight)] = self.move_rules_N(knight)
        i,j = np.where(board_state=="B")
        for bishop in zip(i,j):
            bishops[tuple(bishop)] = self.move_rules_B(bishop)
        i,j = np.where(board_state=="Q")
        for queen in zip(i,j):
            queens[tuple(queen)] = self.move_rules_Q(queen)
        i,j = np.where(board_state=="P")
        for pawn in zip(i,j):
            if threats==False:
                pawns[tuple(pawn)],_ = self.move_rules_P(pawn)
            else:
                _,pawns[tuple(pawn)] = self.move_rules_P(pawn)
        c_dict = {"R": rooks, "N": knights, "B": bishops, "Q": queens, "P": pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values())))); c_list.extend(list(itertools.chain(*list(knights.values())))); 
        c_list.extend(list(itertools.chain(*list(bishops.values())))); c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict

    def move_rules_k(self):
        cb = self.transform_board()
        current_position = np.where(cb=="k")
        i, j = current_position; i,j = i[0],j[0]
        next_positions = []
        c_list, _ = self.possible_W_moves(threats=True)
        for a,b in [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]:
            if 0<=a<=7 and 0<=b<=7:
                if cb[a,b] in [" ","Q","B","N","P","R"] and (a,b) not in c_list:
                    next_positions.append((a,b))
        if self.castle("queenside") == True and self.check_status() == False:
            next_positions.append((0,2))
        if self.castle("kingside") == True and self.check_status() == False:
            next_positions.append((0,6))
        return next_positions

    def possible_B_moves(self,threats=False):
        rooks = {}; knights = {}; bishops = {}; queens = {}; pawns = {};
        board_state = self.transform_board()
        i,j = np.where(board_state=="r")
        for rook in zip(i,j):
            rooks[tuple(rook)] = self.move_rules_r(rook)
        i,j = np.where(board_state=="n")
        for knight in zip(i,j):
            knights[tuple(knight)] = self.move_rules_n(knight)
        i,j = np.where(board_state=="b")
        for bishop in zip(i,j):
            bishops[tuple(bishop)] = self.move_rules_b(bishop)
        i,j = np.where(board_state=="q")
        for queen in zip(i,j):
            queens[tuple(queen)] = self.move_rules_q(queen)
        i,j = np.where(board_state=="p")
        for pawn in zip(i,j):
            if threats==False:
                pawns[tuple(pawn)],_ = self.move_rules_p(pawn)
            else:
                _,pawns[tuple(pawn)] = self.move_rules_p(pawn)
        c_dict = {"r": rooks, "n": knights, "b": bishops, "q": queens, "p": pawns}
        c_list = []
        c_list.extend(list(itertools.chain(*list(rooks.values())))); c_list.extend(list(itertools.chain(*list(knights.values())))); 
        c_list.extend(list(itertools.chain(*list(bishops.values())))); c_list.extend(list(itertools.chain(*list(queens.values()))))
        c_list.extend(list(itertools.chain(*list(pawns.values()))))
        return c_list, c_dict
    
    def move_rules_K(self):
        cb = self.transform_board()
        current_position = np.where(cb=="K")
        i, j = current_position; i,j = i[0],j[0]
        next_positions = []
        c_list, _ = self.possible_B_moves(threats=True)
        for a,b in [(i+1,j),(i-1,j),(i,j+1),(i,j-1),(i+1,j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1)]:
            if 0<=a<=7 and 0<=b<=7:
                if cb[a,b] in [" ","q","b","n","p","r"] and (a,b) not in c_list:
                    next_positions.append((a,b))
        if self.castle("queenside") == True and self.check_status() == False:
            next_positions.append((7,2))
        if self.castle("kingside") == True and self.check_status() == False:
            next_positions.append((7,6))
        return next_positions

    def move_piece(self,initial_position,final_position,promoted_piece="Q"):
        cb = self.transform_board()
        if self.player == 0:
            promoted = False
            i, j = initial_position
            piece = cb[i,j]
            self.current_board = self.current_board.at[i, j].set(self.mapping[" "])
            i, j = final_position
            if piece == "R" and initial_position == (7,0):
                self.R1_move_count += 1
            if piece == "R" and initial_position == (7,7):
                self.R2_move_count += 1
            if piece == "K":
                self.K_move_count += 1
            x, y = initial_position
            if piece == "P":
                if abs(x-i) > 1:
                    self.en_passant = j; self.en_passant_move = self.move_count
                if abs(y-j) == 1 and cb[i,j] == " ": # En passant capture
                    self.current_board = self.current_board.at[i+1,j].set(self.mapping[" "])
                if i == 0 and promoted_piece in ["R","B","N","Q"]:
                    self.current_board = self.current_board.at[i, j].set(self.mapping[promoted_piece])
                    promoted = True
            if promoted == False:
                self.current_board = self.current_board.at[i,j].set(self.mapping[piece])
            self.player = 1
            self.move_count += 1
    
        elif self.player == 1:
            promoted = False
            i, j = initial_position
            piece = cb[i,j]
            self.current_board = self.current_board.at[i,j].set(self.mapping[" "])
            i, j = final_position
            if piece == "r" and initial_position == (0,0):
                self.r1_move_count += 1
            if piece == "r" and initial_position == (0,7):
                self.r2_move_count += 1
            if piece == "k":
                self.k_move_count += 1
            x, y = initial_position
            if piece == "p":
                if abs(x-i) > 1:
                    self.en_passant = j; self.en_passant_move = self.move_count
                if abs(y-j) == 1 and cb[i,j] == " ": # En passant capture
                    self.current_board = self.current_board.at[i-1,j].set(self.mapping[" "])
                if i == 7 and promoted_piece in ["r","b","n","q"]:
                    self.current_board = self.current_board.at[i,j].set(self.mapping[promoted_piece])
                    promoted = True
            if promoted == False:
                self.current_board = self.current_board.at[i-1,j].set(self.mapping[piece])
            self.player = 0
            self.move_count += 1

        else:
            print("Invalid move: ",initial_position,final_position,promoted_piece)

    def update_board(self, i, j, val):
        self.current_board = self.current_board.at[i,j].set(self.mapping[val])
    
    def castle(self,side,inplace=False):
        cb = self.transform_board()
        if self.player == 0 and self.K_move_count == 0:
            if side == "queenside" and self.R1_move_count == 0 and cb[7,1] == " " and cb[7,2] == " "\
                and cb[7,3] == " ":
                if inplace == True:
                    self.update_board(7,0," "); self.update_board(7,3, "R")
                    self.update_board(7,4," "); self.update_board(7,2,"K")
                    self.K_move_count += 1
                    self.player = 1
                return True
            elif side == "kingside" and self.R2_move_count == 0 and cb[7,5] == " " and cb[7,6] == " ":
                if inplace == True:
                    self.update_board(7,7," "); self.update_board(7,5,"R")
                    self.update_board(7,4," "); self.update_board(7,6, "K")
                    self.K_move_count += 1
                    self.player = 1
                return True
        if self.player == 1 and self.k_move_count == 0:
            if side == "queenside" and self.r1_move_count == 0 and cb[0,1] == " " and cb[0,2] == " "\
                and self.current_board[0,3] == " ":
                if inplace == True:
                    self.update_board(0,0," "); self.update_board(0,3,"r")
                    self.update_board(0,4," "); self.update_board(0,2,"k")
                    self.k_move_count += 1
                    self.player = 0
                return True
            elif side == "kingside" and self.r2_move_count == 0 and cb[0,5] == " " and cb[0,6] == " ":
                if inplace == True:
                    self.update_board(0,7," "); self.update_board(0,5,"r")
                    self.update_board(0,4," "); self.update_board(0,6,"k")
                    self.k_move_count += 1
                    self.player = 0
                return True
        return False

    
    def check_status(self):
        cb = self.transform_board()
        if self.player == 0:
            c_list,_ = self.possible_B_moves(threats=True)
            king_position = np.where(cb=="K")
            i, j = king_position
            if (i,j) in c_list:
                return True
        elif self.player == 1:
            c_list,_ = self.possible_W_moves(threats=True)
            king_position = np.where(cb=="k")
            i, j = king_position
            if (i,j) in c_list:
                return True
        return False
    
    def in_check_possible_moves(self):
        self.copy_board = copy.deepcopy(self.transform_board()); self.move_count_copy = self.move_count # backup board state
        self.en_passant_copy = copy.deepcopy(self.en_passant); self.r1_move_count_copy = copy.deepcopy(self.r1_move_count); 
        self.r2_move_count_copy = copy.deepcopy(self.r2_move_count); self.en_passant_move_copy = copy.deepcopy(self.en_passant_move)
        self.k_move_count_copy = copy.deepcopy(self.k_move_count); self.R1_move_count_copy = copy.deepcopy(self.R1_move_count); 
        self.R2_move_count_copy = copy.deepcopy(self.R2_move_count)
        self.K_move_count_copy = copy.deepcopy(self.K_move_count)
        cb = self.transform_board()
        if self.player == 0:
            possible_moves = []
            _, c_dict = self.possible_W_moves()
            current_position = np.where(cb=="K")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["K"] = {(i,j):self.move_rules_K()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.move_piece(initial_pos,final_pos)
                        self.player = 0 # reset board
                        if self.check_status() == False:
                            possible_moves.append([initial_pos, final_pos])
                        self.current_board = self.inverse_board(self.copy_board);
                        self.en_passant = copy.deepcopy(self.en_passant_copy); self.en_passant_move = copy.deepcopy(self.en_passant_move_copy)
                        self.R1_move_count = copy.deepcopy(self.R1_move_count_copy); self.R2_move_count = copy.deepcopy(self.R2_move_count_copy)
                        self.K_move_count = copy.deepcopy(self.K_move_count_copy); self.move_count = self.move_count_copy
            return possible_moves
        if self.player == 1:
            possible_moves = []
            _, c_dict = self.possible_B_moves()
            current_position = np.where(cb=="k")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["k"] = {(i,j):self.move_rules_k()}
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        self.move_piece(initial_pos,final_pos)
                        self.player = 1 # reset board
                        if self.check_status() == False:
                            possible_moves.append([initial_pos, final_pos])
                        self.current_board = self.inverse_board(self.copy_board);
                        self.en_passant = copy.deepcopy(self.en_passant_copy); self.en_passant_move = copy.deepcopy(self.en_passant_move_copy)
                        self.r1_move_count = copy.deepcopy(self.r1_move_count_copy); self.r2_move_count = copy.deepcopy(self.r2_move_count_copy)
                        self.k_move_count = copy.deepcopy(self.k_move_count_copy); self.move_count = self.move_count_copy
            return possible_moves

    def actions(self): # returns all possible actions while not in check: initial_pos,final_pos,underpromote
        acts = []
        cb = self.transform_board(self.current_board)
        if self.player == 0:
            _,c_dict = self.possible_W_moves() # all non-king moves except castling
            current_position = np.where(cb=="K")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["K"] = {(i,j):self.move_rules_K()} # all king moves
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        if key in ["P","p"] and final_pos[0] in [0,7]:
                            for p in ["queen","rook","knight","bishop"]:
                                acts.append([initial_pos,final_pos,p])
                        else:
                            acts.append([initial_pos,final_pos,None])
            actss = []
            for act in acts:  ## after move, check that its not check ownself, else illegal move
                i,f,p = act; b = copy.deepcopy(self)
                b.move_piece(i,f,p)
                b.player = 0
                if b.check_status() == False:
                    actss.append(act)
            return actss
        if self.player == 1:
            _,c_dict = self.possible_B_moves() # all non-king moves except castling
            current_position = np.where(cb=="k")
            i, j = current_position; i,j = i[0],j[0]
            c_dict["k"] = {(i,j):self.move_rules_k()} # all king moves
            for key in c_dict.keys():
                for initial_pos in c_dict[key].keys():
                    for final_pos in c_dict[key][initial_pos]:
                        if key in ["P","p"] and final_pos[0] in [0,7]:
                            for p in ["queen","rook","knight","bishop"]:
                                acts.append([initial_pos,final_pos,p])
                        else:
                            acts.append([initial_pos,final_pos,None])
            actss = []
            for act in acts:  ## after move, check that its not check ownself, else illegal move
                i,f,p = act; b = copy.deepcopy(self)
                b.move_piece(i,f,p)
                b.player = 1
                if b.check_status() == False:
                    actss.append(act)
            return actss

    def encode_board(self, board):
        board_state = self.transform_board(self.current_board)
        encoded = np.zeros([8,8,22]).astype(int)
        encoder_dict = {"R":0, "N":1, "B":2, "Q":3, "K":4, "P":5, "r":6, "n":7, "b":8, "q":9, "k":10, "p":11}
        for i in range(8):
            for j in range(8):
                if board_state[i,j] != " ":
                    encoded[i,j,encoder_dict[board_state[i,j]]] = 1
        if board.player == 1:
            encoded[:,:,12] = 1 # player to move
        if board.K_move_count != 0:
                encoded[:,:,13] = 1 # cannot castle queenside for white
                encoded[:,:,14] = 1 # cannot castle kingside for white
        if board.K_move_count == 0 and board.R1_move_count != 0:
                encoded[:,:,13] = 1
        if board.K_move_count == 0 and board.R2_move_count != 0:
                encoded[:,:,14] = 1
        if board.k_move_count != 0:
                encoded[:,:,15] = 1 # cannot castle queenside for black
                encoded[:,:,16] = 1 # cannot castle kingside for black
        if board.k_move_count == 0 and board.r1_move_count != 0:
                encoded[:,:,15] = 1
        if board.k_move_count == 0 and board.r2_move_count != 0:
                encoded[:,:,16] = 1
        encoded[:,:,17] = board.move_count
        encoded[:,:,18] = board.repetitions_w
        encoded[:,:,19] = board.repetitions_b
        encoded[:,:,20] = board.no_progress_count
        encoded[:,:,21] = board.en_passant
        return jnp.array(encoded, dtype=jnp.int32)

    def decode_board(self, encoded):
        decoded = np.zeros([8,8]).astype(str)
        decoded[decoded == "0.0"] = " "
        decoder_dict = {0:"R", 1:"N", 2:"B", 3:"Q", 4:"K", 5:"P", 6:"r", 7:"n", 8:"b", 9:"q", 10:"k", 11:"p"}
        for i in range(8):
            for j in range(8):
                for k in range(12):
                    if encoded[i,j,k] == 1:
                        decoded[i,j] = decoder_dict[k]
        board = self()
        board.current_board = self.inverse_board(decoded)
        if encoded[0,0,12] == 1:
            board.player = 1
        if encoded[0,0,13] == 1:
            board.R1_move_count = 1
        if encoded[0,0,14] == 1:
            board.R2_move_count = 1
        if encoded[0,0,15] == 1:
            board.r1_move_count = 1
        if encoded[0,0,16] == 1:
            board.r2_move_count = 1
        board.move_count = encoded[0,0,17]
        board.repetitions_w = encoded[0,0,18]
        board.repetitions_b = encoded[0,0,19]
        board.no_progress_count = encoded[0,0,20]
        board.en_passant = encoded[0,0,21]
        return self.inverse_board(board)

    def encode_action(self, board,initial_pos,final_pos,underpromote=None):
        encoded = np.zeros([8,8,73]).astype(int)
        board = self.transform_board(board)
        i, j = initial_pos; x, y = final_pos; dx, dy = x-i, y-j
        piece = board.current_board[i,j]
        if piece in ["R","B","Q","K","P","r","b","q","k","p"] and underpromote in [None,"queen"]: # queen-like moves
            if dx != 0 and dy == 0: # north-south idx 0-13
                if dx < 0:
                    idx = 7 + dx
                elif dx > 0:
                    idx = 6 + dx
            if dx == 0 and dy != 0: # east-west idx 14-27
                if dy < 0:
                    idx = 21 + dy
                elif dy > 0:
                    idx = 20 + dy
            if dx == dy: # NW-SE idx 28-41
                if dx < 0:
                    idx = 35 + dx
                if dx > 0:
                    idx = 34 + dx
            if dx == -dy: # NE-SW idx 42-55
                if dx < 0:
                    idx = 49 + dx
                if dx > 0:
                    idx = 48 + dx
        if piece in ["n","N"]: # Knight moves 56-63
            if (x,y) == (i+2,j-1):
                idx = 56
            elif (x,y) == (i+2,j+1):
                idx = 57
            elif (x,y) == (i+1,j-2):
                idx = 58
            elif (x,y) == (i-1,j-2):
                idx = 59
            elif (x,y) == (i-2,j+1):
                idx = 60
            elif (x,y) == (i-2,j-1):
                idx = 61
            elif (x,y) == (i-1,j+2):
                idx = 62
            elif (x,y) == (i+1,j+2):
                idx = 63
        if piece in ["p", "P"] and (x == 0 or x == 7) and underpromote != None: # underpromotions
            if abs(dx) == 1 and dy == 0:
                if underpromote == "rook":
                    idx = 64
                if underpromote == "knight":
                    idx = 65
                if underpromote == "bishop":
                    idx = 66
            if abs(dx) == 1 and dy == -1:
                if underpromote == "rook":
                    idx = 67
                if underpromote == "knight":
                    idx = 68
                if underpromote == "bishop":
                    idx = 69
            if abs(dx) == 1 and dy == 1:
                if underpromote == "rook":
                    idx = 70
                if underpromote == "knight":
                    idx = 71
                if underpromote == "bishop":
                    idx = 72
        encoded[i,j,idx] = 1
        encoded = encoded.reshape(-1); encoded = np.where(encoded==1)[0][0] #index of action
        return jnp.array(encoded, dtype=jnp.int32)

    def decode_action(self, board,encoded):
        encoded_a = np.zeros([4672]); encoded_a[encoded] = 1; encoded_a = encoded_a.reshape(8,8,73)
        a,b,c = np.where(encoded_a == 1); # i,j,k = i[0],j[0],k[0]
        i_pos, f_pos, prom = [], [], []
        for pos in zip(a,b,c):
            i,j,k = pos
            initial_pos = (i,j)
            promoted = None
            if 0 <= k <= 13:
                dy = 0
                if k < 7:
                    dx = k - 7
                else:
                    dx = k - 6
                final_pos = (i + dx, j + dy)
            elif 14 <= k <= 27:
                dx = 0
                if k < 21:
                    dy = k - 21
                else:
                    dy = k - 20
                final_pos = (i + dx, j + dy)
            elif 28 <= k <= 41:
                if k < 35:
                    dy = k - 35
                else:
                    dy = k - 34
                dx = dy
                final_pos = (i + dx, j + dy)
            elif 42 <= k <= 55:
                if k < 49:
                    dx = k - 49
                else:
                    dx = k - 48
                dy = -dx
                final_pos = (i + dx, j + dy)
            elif 56 <= k <= 63:
                if k == 56:
                    final_pos = (i+2,j-1)
                elif k == 57:
                    final_pos = (i+2,j+1)
                elif k == 58:
                    final_pos = (i+1,j-2)
                elif k == 59:
                    final_pos = (i-1,j-2)
                elif k == 60:
                    final_pos = (i-2,j+1)
                elif k == 61:
                    final_pos = (i-2,j-1)
                elif k == 62:
                    final_pos = (i-1,j+2)
                elif k == 63:
                    final_pos = (i+1,j+2)
            else:
                if k == 64:
                    if board.player == 0:
                        final_pos = (i-1,j)
                        promoted = "R"
                    if board.player == 1:
                        final_pos = (i+1,j)
                        promoted = "r"
                if k == 65:
                    if board.player == 0:
                        final_pos = (i-1,j)
                        promoted = "N"
                    if board.player == 1:
                        final_pos = (i+1,j)
                        promoted = "n"
                if k == 66:
                    if board.player == 0:
                        final_pos = (i-1,j)
                        promoted = "B"
                    if board.player == 1:
                        final_pos = (i+1,j)
                        promoted = "b"
                if k == 67:
                    if board.player == 0:
                        final_pos = (i-1,j-1)
                        promoted = "R"
                    if board.player == 1:
                        final_pos = (i+1,j-1)
                        promoted = "r"
                if k == 68:
                    if board.player == 0:
                        final_pos = (i-1,j-1)
                        promoted = "N"
                    if board.player == 1:
                        final_pos = (i+1,j-1)
                        promoted = "n"
                if k == 69:
                    if board.player == 0:
                        final_pos = (i-1,j-1)
                        promoted = "B"
                    if board.player == 1:
                        final_pos = (i+1,j-1)
                        promoted = "b"
                if k == 70:
                    if board.player == 0:
                        final_pos = (i-1,j+1)
                        promoted = "R"
                    if board.player == 1:
                        final_pos = (i+1,j+1)
                        promoted = "r"
                if k == 71:
                    if board.player == 0:
                        final_pos = (i-1,j+1)
                        promoted = "N"
                    if board.player == 1:
                        final_pos = (i+1,j+1)
                        promoted = "n"
                if k == 72:
                    if board.player == 0:
                        final_pos = (i-1,j+1)
                        promoted = "B"
                    if board.player == 1:
                        final_pos = (i+1,j+1)
                        promoted = "b"
            if self.from_pieces[board.current_board[i,j]] in ["P","p"] and final_pos[0] in [0,7] and promoted == None: # auto-queen promotion for pawn
                if board.player == 0:
                    promoted = "Q"
                else:
                    promoted = "q"
            i_pos.append(initial_pos); f_pos.append(final_pos), prom.append(promoted)
        return i_pos, f_pos, prom