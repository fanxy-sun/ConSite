import sys
import numpy as np
import random


class ContigMap:
    """
    Class for doing mapping.
    Inherited from Inpainting. To update at some point.
    Supports multichain or multiple crops from a single receptor chain.
    Also supports indexing jump (+200) or not, based on contig input.
    Default chain outputs are inpainted chains as A (and B, C etc if multiple chains), and all fragments of receptor chain on the next one (generally B)
    Output chains can be specified. Sequence must be the same number of elements as in contig string
    """

    def __init__(
        self,
        parsed_pdb,
        contigs=None,
        inpaint_seq=None,
        inpaint_str=None,
        length=None,
        ref_idx=None,
        hal_idx=None,
        idx_rf=None,
        inpaint_seq_tensor=None,
        inpaint_str_tensor=None,
        topo=False,
        provide_seq=None,
        inpaint_str_strand=None,
        inpaint_str_helix=None,
        inpaint_str_loop=None
    ):
        """
        初始化 ContigMap 对象。该对象的核心功能是将用户定义的设计规范（通常是字符串），
        翻译成扩散模型可以理解的、精确的、基于索引的映射关系。

        Args:
            parsed_pdb (dict): (必需) 一个包含从输入PDB文件中解析出的所有结构信息的字典，例如坐标、序列和残基索引等。

            contigs (list, optional): (核心参数) 定义蛋白质拓扑结构的contig字符串列表。
                * 多链:
                    * 单链设计: 列表只包含一个字符串，如 `['A1-20/30-40/B5-15']`。
                    * 多链设计: 列表只包含一个字符串，但该字符串内部用**空格**分隔不同的链，如 `['A1-100 20-30']` 用于设计一个两链复合物。这是因为内部解析逻辑会读取列表的第一个元素 (`contigs[0]`) 并用空格分割它。
                    * 错误用法: `['A1-100', '20-30']`。这种写法会导致第二条链 `'20-30'` 被忽略。
                * `contigs` 参数定义了单次设计任务的蓝图，指定了哪些部分是固定的（来自PDB），哪些是新生成的，以及它们的顺序和长度。`inference.num_designs` 参数控制使用同一张蓝图重复进行多少次独立的采样。
                * contigs中的残基索引：
                    * 在 ContigMap.expand_sampled_mask 中，每个形如 A11-30 的片段会被直接解析成 ('A', 11)…('A', 30)，随后用来到 parsed_pdb 里取对应残基。也就是说，contigs中的残基索引必须和 PDB 文件里残基的 原始 resSeq 编号 对得上，而不是脚本重新编号的索引。
                    * 连续编号、单链示例：如果输入 PDB 的 A 链从 1 到 300 都存在并连续，就写 ['A1-300/0']；/0 表示这一整段是 receptor（固定上下文）。
                    * 编号整体平移：若 PDB 从 11 开始到 310 结束，就写 ['A11-310/0']。不要强行改成 1-300，因为内部会去找 ('A', 1) 这样的残基，结果找不到。
                    * 存在缺口的情况：当 A 链有缺口，比如只存在 11–50 和 60–120，可拆成多个连续片段：['A11-50/A60-120/0']，这样 expand_sampled_mask 会自动在 50→60 之间插入 9 的链距信息。

            inpaint_seq (list, optional): 指定需要进行“序列修复”(sequence inpainting)的区域。
                                          在这些区域，骨架结构是固定的(来自PDB)，仅氨基酸序列由模型重新设计。
                                          此过程不涉及从噪声生成结构，而是基于固定的3D坐标寻找最优序列。
                                          示例: ['A15-18'] 表示A链15-18号残基的骨架被保留，但序列由模型重新生成。

            inpaint_str (list, optional): 指定需要进行“结构修复”(structure inpainting)的区域。
                                          模型会忽略这些区域的原始结构和序列，将其作为噪声进行初始化。
                                          **区别于无条件生成**：此过程是在周围固定结构（上下文）的约束下，
                                          从噪声中有条件地生成一个能与上下文完美衔接的结构片段，
                                          而非完全从纯噪声开始生成整个蛋白质。
                                          示例: ['A1-5'] 表示A链的1-5号残基的结构和序列都将被完全重新生成。

            length (str, optional): (旧版参数) 指定生成部分的总长度或长度范围。现在更推荐在
                                    `contigs`字符串中直接定义长度。
                                    示例: '100-120'。

            provide_seq (list, optional): 在“部分扩散”模式下，为新生成的区域提供序列模板。
                                          模型将基于此序列来生成相应的结构。
                                          示例: ['10-20'] 表示新生成链的第10到20个残基将使用这里提供的序列。

            inpaint_str_helix (list, optional): 在`inpaint_str`或新生成的区域中，约束一个片段必须形成α螺旋。
            inpaint_str_strand (list, optional): 约束一个片段必须形成β折叠。
            inpaint_str_loop (list, optional): 约束一个片段必须形成环区（loop）。
                                               示例: 当 `inpaint_str=['10-30']` 时, 设置 `inpaint_str_helix=['15-25']`
                                               会强制在重新设计的10-30区域中，15-25号残基形成α螺旋。

            ref_idx (list, optional): (高级) 直接提供参考索引列表，绕过`contigs`字符串解析，用于精确的底层控制。
            hal_idx (list, optional): (高级) 直接提供“幻觉”（生成）索引列表。
            idx_rf (list, optional): (高级) 直接提供给底层RoseTTAFold模型使用的索引列表。

            inpaint_seq_tensor (torch.Tensor, optional): (内部使用) `inpaint_seq`的张量表示。
            inpaint_str_tensor (torch.Tensor, optional): (内部使用) `inpaint_str`的张量表示。
            topo (bool, optional): (内部使用) 拓扑模式的标志。
        """
        # sanity checks
        if contigs is None and ref_idx is None:
            sys.exit("Must either specify a contig string or precise mapping")
        if idx_rf is not None or hal_idx is not None or ref_idx is not None:
            if idx_rf is None or hal_idx is None or ref_idx is None:
                sys.exit(
                    "If you're specifying specific contig mappings, the reference and output positions must be specified, AND the indexing for RoseTTAFold (idx_rf)"
                )

        self.chain_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # self.length 表示新生成片段的长度范围（即需要模型生成的残基数量区间），用于约束 contig 设计的总长度。
        if length is not None:
            if "-" not in length:
                self.length = [int(length), int(length) + 1]
            else:
                self.length = [int(length.split("-")[0]), int(length.split("-")[1]) + 1]
        else:
            self.length = None
        self.ref_idx = ref_idx
        self.hal_idx = hal_idx
        self.idx_rf = idx_rf

        parse_inpaint = lambda x: "/".join(x).split("/") if x is not None else None
        self.inpaint_seq = parse_inpaint(inpaint_seq)
        self.inpaint_str = parse_inpaint(inpaint_str)

        self.inpaint_str_helix=parse_inpaint(inpaint_str_helix)
        self.inpaint_str_strand=parse_inpaint(inpaint_str_strand)
        self.inpaint_str_loop=parse_inpaint(inpaint_str_loop)

        self.inpaint_seq_tensor = inpaint_seq_tensor
        self.inpaint_str_tensor = inpaint_str_tensor
        self.parsed_pdb = parsed_pdb
        self.topo = topo
        if ref_idx is None:
            # using default contig generation, which outputs in rosetta-like format
            self.contigs = contigs
            (
                self.sampled_mask,
                self.contig_length,
                self.n_inpaint_chains,
            ) = self.get_sampled_mask()
            self.receptor_chain = self.chain_order[self.n_inpaint_chains]
            (
                self.receptor,
                self.receptor_hal,
                self.receptor_rf,
                self.inpaint,
                self.inpaint_hal,
                self.inpaint_rf,
            ) = self.expand_sampled_mask()

            # 数据结构: 一个由元组 (chain_id, residue_id) 组成的列表。
            # 核心作用: 定义了模型中每个残基的“来源”。它建立了从模型内部的一维序列到输入PDB文件坐标的映射关系。
            #       * self.inpaint: 包含了所有“inpaint”区域（即需要模型生成或修复的部分）的来源信息。
            #               * 如果是一个固定的基序（Motif），元组会是 ('A', 33)，表示这个残基来源于输入PDB的A链第33号残基。
            #               * 如果是一个全新生成的部分（长度由 20-30 这样的字符串定义），元组会是 ('_', '_')，表示这个残基在输入PDB中没有来源，是“无中生有”的。
            #       * self.receptor: 包含了所有“receptor”区域（即作为固定上下文的结构部分）的来源信息，例如 ('B', 5)。
            # 拼接结果 (self.ref): 这是一个完整的、有序的列表，描述了最终要生成的蛋白质中，每一个残基分别对应输入PDB的哪个原子坐标。它是构建初始结构 xyz_t 和序列 seq 的直接依据。
            self.ref = self.inpaint + self.receptor

            # 数据结构: 一个由元组 (new_chain_id, new_residue_id) 组成的列表。
            # 核心作用: 定义了最终输出PDB文件中的“身份”。它为模型中的每一个残基预先分配好了在输出文件中的链ID和残基编号。
            #       * self.inpaint_hal: 为所有“inpaint”链段分配新的、从头开始的链ID和连续的残基编号。例如，第一个inpaint链会被标记为 [('A', 1), ('A', 2), ...]。
            #       * self.receptor_hal: 为所有“receptor”链段分配新的链ID和编号。通常，它的链ID会接在inpaint链之后（如'B', 'C'等），并且其残基编号可能会被整体偏移，以避免与inpaint链重叠。
            # 拼接结果 (self.hal): 这是一个完整的、有序的列表，描绘了最终生成的蛋白质PDB文件的蓝图。它决定了输出的 *.pdb 文件中每个原子的链和残基信息。
            self.hal = self.inpaint_hal + self.receptor_hal

            # 数据结构: 一个由整数组成的列表。
            # 核心作用: 定义了RoseTTAFold模型内部使用的、包含拓扑信息的一维索引。这是最特殊的一个索引，因为它通过引入“索引跳跃”（Index Jumps）来向模型传递链的连接性信息。
            #       * self.inpaint_rf: 为“inpaint”部分生成索引，通常从0开始连续递增。
            #       * self.receptor_rf: 为“receptor”部分生成索引。
            #       * 关键机制: 在拼接 inpaint_rf 和 receptor_rf 之前，以及在处理多条链或不连续片段时，expand_sampled_mask 函数会有策略地在索引中插入一个巨大的数字（通常是200）。例如，[..., 59, 60, 261, 262, ...]。这个 60 -> 261 的巨大跳跃就是一个明确的信号，告诉模型这两个残基在空间上可能很近，但在共价键上是断开的。这对于模型正确理解多链复合物或不连续结构至关重要。
            # 拼接结果 (self.rf): 这是一个最终输入给 RoseTTAFoldModule 的 idx 张量。它不仅告诉模型每个残基的位置，还通过索引的连续性与跳跃，隐式地编码了蛋白质的拓扑结构。
            self.rf = self.inpaint_rf + self.receptor_rf
        else:
            # specifying precise mappings
            self.ref = ref_idx
            self.hal = hal_idx
            self.rf = idx_rf

        # self.mask_1d掩码标识了模型中哪些残基是有结构参考的 (True)，哪些是完全从头生成的 (False)。它在后续步骤中被用来区分处理这两种不同来源的残基。
        self.mask_1d = [False if i == ("_", "_") else True for i in self.ref]

        # self.inpaint_seq最终是一个布尔型 (boolean) 的 NumPy 数组，长度等于蛋白质总长度 L。标记哪些残基的氨基酸序列是“固定的”掩码。
        #       * self.inpaint_seq[i] == True: 意味着第 i 个残基的氨基酸序列是固定的，模型不应该改变它。这通常用于 PDB 中已知的基序（motif）或骨架区域。
        #       * self.inpaint_seq[i] == False: 意味着第 i 个残基的氨基酸序列是需要模型生成/预测的。这是“序列待修复”的区域。
        if self.inpaint_seq_tensor is None:
            if self.inpaint_seq is not None:
                self.inpaint_seq = self.get_inpaint_seq_str(self.inpaint_seq)
            else:
                self.inpaint_seq = np.array(
                    [True if i != ("_", "_") else False for i in self.ref]
                )
        else:
            self.inpaint_seq = self.inpaint_seq_tensor

        # self.inpaint_str最终是一个布尔型 (boolean) 的 NumPy 数组，长度等于蛋白质总长度 L。标记哪些残基的原子坐标是“固定的”掩码。
        #       * self.inpaint_str[i] == True: 意味着第 i 个残基的三维结构是固定的，不参与扩散过程（即不加噪声，也不去噪）。这通常是作为“脚手架”或“上下文”的区域。
        #       * self.inpaint_str[i] == False: 意味着第 i 个残基的三维结构是需要模型从噪声中生成的。这是真正的“结构待修复”或“待生成”区域。
        if self.inpaint_str_tensor is None:
            if self.inpaint_str is not None:
                self.inpaint_str = self.get_inpaint_seq_str(self.inpaint_str)
            else:
                self.inpaint_str = np.array(
                    [True if i != ("_", "_") else False for i in self.ref]
                )
        else:
            self.inpaint_str = self.inpaint_str_tensor


        # get 0-indexed input/output (for trb file)
        (
            self.ref_idx0,
            self.hal_idx0,
            self.ref_idx0_inpaint,
            self.hal_idx0_inpaint,
            self.ref_idx0_receptor,
            self.hal_idx0_receptor,
        ) = self.get_idx0()
        self.con_ref_pdb_idx = [i for i in self.ref if i != ("_", "_")]

        # Handle provide seq. This is zero-indexed, and used only for partial diffusion
        # --provide_seq 的意思是“在这些新生成的区域，请使用我提供的序列”。因此，模型在这些位置不需要进行序列修复（inpaint a sequence），而是应该固定（keep）用户提供的序列。将 inpaint_seq 设置为 True 正是“固定序列”的意思。
        if provide_seq is not None:
            for i in provide_seq:
                if "-" in i:
                    self.inpaint_seq[
                        int(i.split("-")[0]) : int(i.split("-")[1]) + 1
                    ] = True
                else:
                    self.inpaint_seq[int(i)] = True

        """
        We have now added the ability to specify the secondary structure of provided sequence.
        This is described in Liu et al., 2024
        https://www.biorxiv.org/content/10.1101/2024.07.16.603789v1
        This is for the case that e.g. you have a sequence, but don't know the structure (like an IDR), but
        want to specify the secondary structure of this sequence.
        Making this compatible with the contigmap object allows all the variable length stuff to be handled.
        
        The logic:
        Secondary structure is provided at the command line, using the following three flags:
            inpaint_str_helix
            inpaint_str_strand
            inpaint_str_loop
        
        These are so named because they pertain to the region of the input pdb that you have applied inpaint_str to
        In other words, any part of the input protein you are masking the structure of, you can specify the secondary structure of.
        However, you can't specify the secondary structure of a region you're not applying inpaint_str to, as this doesn't make sense.
        """

        if any(x is not None for x in (inpaint_str_helix, inpaint_str_strand, inpaint_str_loop)):
            self.ss_spec={}
            order=['helix','strand','loop']
            for idx, i in enumerate([inpaint_str_helix, inpaint_str_strand, inpaint_str_loop]):
                if i is not None:
                    self.ss_spec[order[idx]] = ~self.get_inpaint_seq_str(i, ss=True)
                else:
                    self.ss_spec[order[idx]] = np.zeros(len(self.inpaint_seq), dtype=bool)
            # some sensible checks
            for key, mask in self.ss_spec.items():
                assert sum(mask*self.inpaint_str) == 0, f"You've specified {key} residues that are not structure-masked with inpaint_str. This doesn't really make sense."
            stack=np.vstack([mask for mask in self.ss_spec.values()])
            assert np.max(np.sum(stack, axis=0)) == 1, "You've given multiple secondary structure assignations to an input residue. This doesn't make sense."

    def get_sampled_mask(self):
        """
        从 contig 字符串中采样一个具体的、长度固定的 "mask"。

        在 get_sampled_mask 中，凡是不满足“所有子段都以字母开头且最后一段是 0”的 contig 片段都会走 “inpaint 链” 分支（视为inpaint 区域）。
        这一分支不仅把纯数字的采样区段当成 inpaint，还会将形如 A1-10、B5-15 这样的字母开头子段加入。随后在 expand_sampled_mask（约第 392-443 行）里，进入 inpaint 分支的字母片段会被追加到 inpaint 列表，并保留原始 PDB 索引；真正需要新生成的区间才会变成 ('_','_') 占位。
        因此，固定 motif（用 PDB 残基号描述的字母片段）被视为 inpaint 链的一部分：它们在扩散过程中提供约束，坐标保持不变，但仍被归入 inpaint 相关的数据结构。

        这个函数的核心作用是处理 self.contigs 列表中的第一个字符串元素。它会将这个字符串首先按空格分割成多个部分（代表不同的链或链组），然后将其中表示长度范围的片段（如 "50-100"）通过随机采样，转换成一个确定长度的片段（如 "75-75"）。
        最终，它会生成一个所有长度都已确定的、用于单次生成任务的具体 "蓝图"。

        注意：此函数只处理 `self.contigs` 列表的第一个元素 (`self.contigs[0]`)，并使用空格作为该字符串内部分隔不同链定义的分隔符。这是一种特殊的用法。在标准的用法中，`contigs` 列表的每个元素代表一条独立的链，此时应避免在字符串内使用空格。

        对于活性位点预测任务，假设要预测一个只有A链、300个残基的蛋白质的活性位点，输入的contigs是['A1-300/0']，执行get_sampled_mask函数，会得到：
            * sampled_mask = ['A1-300/0']
            * sampled_mask_length = 0
            * inpaint_chains = 0

        Returns:
            tuple: 包含三个元素的元组:
            - sampled_mask (list): 一个列表，其中每个元素都是一个长度被完全实例化的 contig 字符串。任何在原始 contig 中的长度范围（例如 '30-40'）都已被一个具体的、随机采样的长度（例如 '35-35'）所替代。
                示例: 如果 `self.contigs[0]` 是 "A1-10/20-30/B1-5 10-15"，一个可能的 `sampled_mask` 返回值是 ['A1-10/25-25/B1-5', '12-12']。

            - sampled_mask_length (int): `sampled_mask` 中所有inpaint 区域片段长度的总和。只累加 inpaint 区域 的残基数，不包含 receptor 区域。
                示例: 对于上面的 `sampled_mask` ['A1-10/25-25/B1-5', '12-12']，其总长度为 10 (来自 A1-10) + 25 (采样得到) + 5 (来自 B1-5) + 12 (采样得到) = 52。因此 `sampled_mask_length` 返回 52。

            - inpaint_chains (int): 需要进行 inpainting 或从头生成的链的数量。在 `self.contigs[0]` 字符串中，每个以空格分隔的、且包含待生成片段（非纯 PDB 区域）的元素都被视为一个 "inpaint_chain"。
                示例: 对于 "A1-10/20-30/B1-5 10-15"，有两个部分需要生成，因此 `inpaint_chains` 返回 2。对于 "A1-20/50-60/B1-10"，只有一个部分需要生成，因此 `inpaint_chains` 返回 1。
        """
        length_compatible = False
        count = 0
        while length_compatible is False:
            inpaint_chains = 0
            contig_list = self.contigs[0].strip().split()
            sampled_mask = []
            sampled_mask_length = 0
            # allow receptor chain to be last in contig string
            if all([i[0].isalpha() for i in contig_list[-1].split("/")]):
                contig_list[-1] = f"{contig_list[-1]}/0"
            for con in contig_list:
                if (
                    all([i[0].isalpha() for i in con.split("/")[:-1]])
                    and con.split("/")[-1] == "0"
                ) or self.topo is True:
                    # receptor chain
                    sampled_mask.append(con)
                else:
                    inpaint_chains += 1
                    # chain to be inpainted. These are the only chains that count towards the length of the contig
                    subcons = con.split("/")
                    subcon_out = []
                    for subcon in subcons:
                        if subcon[0].isalpha():
                            subcon_out.append(subcon)
                            if "-" in subcon:
                                sampled_mask_length += (
                                    int(subcon.split("-")[1])
                                    - int(subcon.split("-")[0][1:])
                                    + 1
                                )
                            else:
                                sampled_mask_length += 1

                        else:
                            if "-" in subcon:
                                length_inpaint = random.randint(
                                    int(subcon.split("-")[0]), int(subcon.split("-")[1])
                                )
                                subcon_out.append(f"{length_inpaint}-{length_inpaint}")
                                sampled_mask_length += length_inpaint
                            elif subcon == "0":
                                subcon_out.append("0")
                            else:
                                length_inpaint = int(subcon)
                                subcon_out.append(f"{length_inpaint}-{length_inpaint}")
                                sampled_mask_length += int(subcon)
                    sampled_mask.append("/".join(subcon_out))
            # check length is compatible
            if self.length is not None:
                if (
                    sampled_mask_length >= self.length[0]
                    and sampled_mask_length < self.length[1]
                ):
                    length_compatible = True
            else:
                length_compatible = True
            count += 1
            if count == 100000:  # contig string incompatible with this length
                sys.exit("Contig string incompatible with --length range")
        return sampled_mask, sampled_mask_length, inpaint_chains

    def expand_sampled_mask(self):
        """
        将采样后的contig掩码解析为供模型使用的、包含多种详细信息的索引映射。

        此函数将来自 `self.sampled_mask` 的抽象contig字符串，转换为扩散模型不同组件可以理解的具体残基索引。它区分了“receptor”（来自PDB的固定部分）和“inpaint”（生成/支架化的部分），并为它们在三种不同的坐标系中生成索引：
            * 原始PDB索引。
            * 用于最终输出模型的“幻觉”（Hallucinated）索引。
            * 包含链断裂信息的RoseTTAFold特定索引 (`_rf`)。

        对于之前的活性位点预测例子：要预测一个只有A链、300个残基的蛋白质的活性位点，输入的contigs是['A1-300/0']，执行get_sampled_mask函数，会得到：
            * sampled_mask = ['A1-300/0']
            * sampled_mask_length = 0
            * inpaint_chains = 0
        执行expand_sampled_mask后，输出
            * receptor = [('A', 1), ('A', 2), ..., ('A', 300)]
            * receptor_hal = [('A', 1), ('A', 2), ..., ('A', 300)]。RFdiffusion 输出结构中 receptor 链的链 ID 与新编号。因为没有 inpaint 链，所以 receptor 也用 'A'，编号连续 1-300。
            * receptor_rf = [200, 201, 202, ..., 499]。供 RoseTTAFold 的索引序列。开头加 200，告诉模型这条链与 inpaint 空间隔开。
            * inpaint = []。此次没有需要生成的 inpaint 片段。
            * inpaint_hal = []。对应 inpaint 的输出链编号，同样为空。
            * inpaint_rf = []。Inpaint 的 RF 索引为空数组。

        Returns:
            tuple: 一个包含六个列表的元组：
            - receptor (list): 一个元组列表 `(chain_id, residue_id)`，包含所有属于结构中固定的“receptor”部分的残基，使用原始PDB的链ID和残基编号。示例: [('A', 1), ('A', 2), ...]
            - receptor_hal (list): 一个元组列表 `(new_chain_id, new_residue_id)`，用于“receptor”残基，这些索引被重新编号以适应最终“幻觉”（输出）的蛋白质结构。链ID被重新分配，残基编号被调整为连续或偏移。示例: [('B', 121), ('B', 122), ...]
            - receptor_rf (list): 一个整数列表，代表RoseTTAFold模型所见的“receptor”残基的最终索引。此列表包含大的整数跳跃（如+200），以表示不连续片段或不同链之间的链断裂。示例: [200, 201, 202, 403, 404, ...]

            - inpaint (list): 一个元组列表 `(chain_id, residue_id)`，包含所有“inpaint”区域的残基。对于固定的基序，它包含原始PDB信息（如('A', 15)）。对于新生成的片段，它使用占位符 `('_', '_')`。示例: [('A', 15), ('_', '_'), ('_', '_'), ...]
            - inpaint_hal (list): 一个元组列表 `(new_chain_id, new_residue_id)`，用于“inpaint”残基，这些索引被重新编号以适应最终输出结构。这为所有新设计的链提供了一个完整、有序的表示。示例: [('A', 1), ('A', 2), ('A', 3), ...]
            - inpaint_rf (list): 一个整数列表，代表RoseTTAFold模型所见的“inpaint”残基的最终索引，包含链断裂的跳跃。这通常从0开始。示例: [0, 1, 2, 203, 204, ...]
        """
        chain_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        receptor = []
        inpaint = []
        receptor_hal = []
        inpaint_hal = []
        receptor_idx = 1
        inpaint_idx = 1
        inpaint_chain_idx = -1
        receptor_chain_break = []
        inpaint_chain_break = []
        for con in self.sampled_mask:
            if (
                all([i[0].isalpha() for i in con.split("/")[:-1]])
                and con.split("/")[-1] == "0"
            ) or self.topo is True:
                # receptor chain
                subcons = con.split("/")[:-1]
                assert all(
                    [i[0] == subcons[0][0] for i in subcons]
                ), "If specifying fragmented receptor in a single block of the contig string, they MUST derive from the same chain"
                assert all(
                    int(subcons[i].split("-")[0][1:])
                    < int(subcons[i + 1].split("-")[0][1:])
                    for i in range(len(subcons) - 1)
                ), "If specifying multiple fragments from the same chain, pdb indices must be in ascending order!"
                for idx, subcon in enumerate(subcons):
                    ref_to_add = [
                        (subcon[0], i)
                        for i in np.arange(
                            int(subcon.split("-")[0][1:]), int(subcon.split("-")[1]) + 1
                        )
                    ]
                    receptor.extend(ref_to_add)
                    receptor_hal.extend(
                        [
                            (self.receptor_chain, i)
                            for i in np.arange(
                                receptor_idx, receptor_idx + len(ref_to_add)
                            )
                        ]
                    )
                    receptor_idx += len(ref_to_add)
                    if idx != len(subcons) - 1:
                        idx_jump = (
                            int(subcons[idx + 1].split("-")[0][1:])
                            - int(subcon.split("-")[1])
                            - 1
                        )
                        receptor_chain_break.append(
                            (receptor_idx - 1, idx_jump)
                        )  # actual chain break in pdb chain
                    else:
                        receptor_chain_break.append(
                            (receptor_idx - 1, 200)
                        )  # 200 aa chain break
            else:
                inpaint_chain_idx += 1
                for subcon in con.split("/"):
                    if subcon[0].isalpha():
                        ref_to_add = [
                            (subcon[0], i)
                            for i in np.arange(
                                int(subcon.split("-")[0][1:]),
                                int(subcon.split("-")[1]) + 1,
                            )
                        ]
                        inpaint.extend(ref_to_add)
                        inpaint_hal.extend(
                            [
                                (chain_order[inpaint_chain_idx], i)
                                for i in np.arange(
                                    inpaint_idx, inpaint_idx + len(ref_to_add)
                                )
                            ]
                        )
                        inpaint_idx += len(ref_to_add)

                    else:
                        inpaint.extend([("_", "_")] * int(subcon.split("-")[0]))
                        inpaint_hal.extend(
                            [
                                (chain_order[inpaint_chain_idx], i)
                                for i in np.arange(
                                    inpaint_idx, inpaint_idx + int(subcon.split("-")[0])
                                )
                            ]
                        )
                        inpaint_idx += int(subcon.split("-")[0])
                inpaint_chain_break.append((inpaint_idx - 1, 200))

        if self.topo is True or inpaint_hal == []:
            receptor_hal = [(i[0], i[1]) for i in receptor_hal]
        else:
            receptor_hal = [
                (i[0], i[1] + inpaint_hal[-1][1]) for i in receptor_hal
            ]  # rosetta-like numbering
        # get rf indexes, with chain breaks
        inpaint_rf = np.arange(0, len(inpaint))
        receptor_rf = np.arange(len(inpaint) + 200, len(inpaint) + len(receptor) + 200)
        for ch_break in inpaint_chain_break[:-1]:
            receptor_rf[:] += 200
            inpaint_rf[ch_break[0] :] += ch_break[1]
        for ch_break in receptor_chain_break[:-1]:
            receptor_rf[ch_break[0] :] += ch_break[1]

        return (
            receptor,
            receptor_hal,
            receptor_rf.tolist(),
            inpaint,
            inpaint_hal,
            inpaint_rf.tolist(),
        )

    def get_inpaint_seq_str(self, inpaint_s, ss=False):
        '''
        function to generate inpaint_str or inpaint_seq masks specific to this contig
        '''
        if not ss:
            s_mask = np.copy(self.mask_1d)
        else:
            s_mask= np.ones(len(self.mask_1d), dtype=bool)
        inpaint_s_list = []
        for i in inpaint_s:
            if "-" in i:
                inpaint_s_list.extend(
                    [
                        (i[0], p)
                        for p in range(
                            int(i.split("-")[0][1:]), int(i.split("-")[1]) + 1
                        )
                    ]
                )
            else:
                inpaint_s_list.append((i[0], int(i[1:])))
        for res in inpaint_s_list:
            if res in self.ref:
                s_mask[self.ref.index(res)] = False  # mask this residue

        return np.array(s_mask)

    def get_idx0(self):
        """
        生成从原始PDB索引到模型内部索引的映射。

        "ref" (reference) 指的是原始PDB中的残基，"hal" (hallucination) 指的是这些残基在模型输入张量中的新位置。
        所有返回的索引都是0-based的。

        Args:
            无

        Returns:
            tuple: 包含六个列表的元组。
                - ref_idx0 (list): `self.ref` (完整序列)中所有来源于PDB的残基在`parsed_pdb`中的索引。
                - hal_idx0 (list): `self.ref` (完整序列)中所有来源于PDB的残基在自身序列中的索引。
                - ref_idx0_inpaint (list): `self.inpaint` (待修复/生成区域)中来源于PDB的残基在`parsed_pdb`中的索引。
                - hal_idx0_inpaint (list): `self.inpaint` (待修复/生成区域)中来源于PDB的残基在自身序列中的索引。
                - ref_idx0_receptor (list): `self.receptor` (受体/固定区域)中来源于PDB的残基在`parsed_pdb`中的索引。
                - hal_idx0_receptor (list): `self.receptor` (受体/固定区域)中来源于PDB的残基在自身序列中的索引。

        Example:
            **场景设定**:
            - 输入PDB: 单链A，长度100。`parsed_pdb['pdb_idx']` 为 `[('A', 1), ..., ('A', 100)]`。
            - Contig指令: `contigs='A10-20/11/A60-70'` (保留A10-20和A60-70，中间生成11个残基)。

            **ContigMap内部状态**:
            - `self.ref` (总序列, L=33): `[('A',10),...,('A',20), ('_','_'),...,('_','_'), ('A',60),...,('A',70)]`
            - `self.inpaint` (生成区, L=11): `[('_','_'),...,('_','_')]`
            - `self.receptor` (固定区, L=22): `[('A',10),...,('A',20), ('A',60),...,('A',70)]`

            **函数输出**:
            - `ref_idx0`: `[9, ..., 19, 59, ..., 69]`
              (A10-20和A60-70在PDB中的0-based索引)
            - `hal_idx0`: `[0, ..., 10, 22, ..., 32]`
              (A10-20和A60-70在`self.ref`中的0-based索引)
            - `ref_idx0_inpaint`: `[]`
              (inpaint区域没有来自PDB的残基)
            - `hal_idx0_inpaint`: `[]`
              (inpaint区域没有来自PDB的残基)
            - `ref_idx0_receptor`: `[9, ..., 19, 59, ..., 69]`
              (receptor区域的残基在PDB中的0-based索引)
            - `hal_idx0_receptor`: `[0, ..., 10, 11, ..., 21]`
              (receptor区域的残基在`self.receptor`中的0-based索引)
        """
        ref_idx0 = []
        hal_idx0 = []
        ref_idx0_inpaint = []
        hal_idx0_inpaint = []
        ref_idx0_receptor = []
        hal_idx0_receptor = []
        for idx, val in enumerate(self.ref):
            if val != ("_", "_"):
                assert val in self.parsed_pdb["pdb_idx"], f"{val} is not in pdb file!"
                hal_idx0.append(idx)
                ref_idx0.append(self.parsed_pdb["pdb_idx"].index(val))
        for idx, val in enumerate(self.inpaint):
            if val != ("_", "_"):
                hal_idx0_inpaint.append(idx)
                ref_idx0_inpaint.append(self.parsed_pdb["pdb_idx"].index(val))
        for idx, val in enumerate(self.receptor):
            if val != ("_", "_"):
                hal_idx0_receptor.append(idx)
                ref_idx0_receptor.append(self.parsed_pdb["pdb_idx"].index(val))

        return (
            ref_idx0,
            hal_idx0,
            ref_idx0_inpaint,
            hal_idx0_inpaint,
            ref_idx0_receptor,
            hal_idx0_receptor,
        )

    def get_mappings(self):
        mappings = {}

        # con_ref_pdb_idx：所有 inpaint 区域中真实来源于输入PDB的残基的 (chain, res_id) 元组列表。用于追踪哪些 inpaint 区域实际上是“修复”而不是“全新生成”。
        mappings["con_ref_pdb_idx"] = [i for i in self.inpaint if i != ("_", "_")]

        # con_hal_pdb_idx：所有 inpaint 区域中真实来源于PDB的残基在输出结构中的新链ID和新残基编号。用于输出PDB时，定位这些修复残基在新结构中的位置。
        mappings["con_hal_pdb_idx"] = [
            self.inpaint_hal[i]
            for i in range(len(self.inpaint_hal))
            if self.inpaint[i] != ("_", "_")
        ]
        mappings["con_ref_idx0"] = np.array(self.ref_idx0_inpaint)
        mappings["con_hal_idx0"] = np.array(self.hal_idx0_inpaint)
        if self.inpaint != self.ref:
            mappings["complex_con_ref_pdb_idx"] = [
                i for i in self.ref if i != ("_", "_")
            ]
            mappings["complex_con_hal_pdb_idx"] = [
                self.hal[i] for i in range(len(self.hal)) if self.ref[i] != ("_", "_")
            ]
            mappings["receptor_con_ref_pdb_idx"] = [
                i for i in self.receptor if i != ("_", "_")
            ]
            mappings["receptor_con_hal_pdb_idx"] = [
                self.receptor_hal[i]
                for i in range(len(self.receptor_hal))
                if self.receptor[i] != ("_", "_")
            ]
            mappings["complex_con_ref_idx0"] = np.array(self.ref_idx0)
            mappings["complex_con_hal_idx0"] = np.array(self.hal_idx0)
            mappings["receptor_con_ref_idx0"] = np.array(self.ref_idx0_receptor)
            mappings["receptor_con_hal_idx0"] = np.array(self.hal_idx0_receptor)
        mappings["inpaint_str"] = self.inpaint_str
        mappings["inpaint_seq"] = self.inpaint_seq
        mappings["sampled_mask"] = self.sampled_mask
        mappings["mask_1d"] = self.mask_1d
        return mappings
