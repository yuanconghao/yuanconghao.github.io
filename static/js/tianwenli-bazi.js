var BAZI_WUXING_NAMES = ["金", "水", "木", "火", "土"];
var BAZI_SHISHEN_SHORT = ["印", "枭", "比", "劫", "伤", "食", "财", "才", "官", "杀"];
var BAZI_SHISHEN_FULL = ["正印", "偏印", "比肩", "劫财", "伤官", "食神", "正财", "偏财", "正官", "七杀"];
var BAZI_CHANGSHENG_SHORT = ["长", "沐", "冠", "建", "帝", "衰", "病", "死", "墓", "绝", "胎", "养"];
var BAZI_NAYIN = [
	"海中金", "炉中火", "大林木", "路旁土", "剑锋金", "山头火",
	"涧下水", "城头土", "白蜡金", "杨柳木", "泉中水", "屋上土",
	"霹雳火", "松柏木", "长流水", "沙中金", "山下火", "平地木",
	"壁上土", "金箔金", "佛灯火", "天河水", "大驿土", "钗钏金",
	"桑柘木", "大溪水", "沙中土", "天上火", "石榴木", "大海水"
];
var BAZI_EMPTY_BY_XUN = [
	["戌", "亥"],
	["申", "酉"],
	["午", "未"],
	["辰", "巳"],
	["寅", "卯"],
	["子", "丑"]
];
var BAZI_DAY_STEM_STAGE_START = [1, 6, 10, 9, 10, 9, 7, 0, 4, 3];
var BAZI_ZHI_HIDDEN_STEMS = {
	"子": [{ gan: "癸", weight: 100 }],
	"丑": [{ gan: "己", weight: 60 }, { gan: "癸", weight: 30 }, { gan: "辛", weight: 10 }],
	"寅": [{ gan: "甲", weight: 60 }, { gan: "丙", weight: 30 }, { gan: "戊", weight: 10 }],
	"卯": [{ gan: "乙", weight: 100 }],
	"辰": [{ gan: "戊", weight: 60 }, { gan: "乙", weight: 30 }, { gan: "癸", weight: 10 }],
	"巳": [{ gan: "丙", weight: 60 }, { gan: "戊", weight: 30 }, { gan: "庚", weight: 10 }],
	"午": [{ gan: "丁", weight: 70 }, { gan: "己", weight: 30 }],
	"未": [{ gan: "己", weight: 60 }, { gan: "乙", weight: 30 }, { gan: "丁", weight: 10 }],
	"申": [{ gan: "庚", weight: 60 }, { gan: "壬", weight: 30 }, { gan: "戊", weight: 10 }],
	"酉": [{ gan: "辛", weight: 100 }],
	"戌": [{ gan: "戊", weight: 60 }, { gan: "辛", weight: 30 }, { gan: "丁", weight: 10 }],
	"亥": [{ gan: "壬", weight: 70 }, { gan: "甲", weight: 30 }]
};
var BAZI_YEAR_SHENS = {
	"孤辰": { "子": "寅", "丑": "寅", "寅": "巳", "卯": "巳", "辰": "巳", "巳": "申", "午": "申", "未": "申", "申": "亥", "酉": "亥", "戌": "亥", "亥": "寅" },
	"寡宿": { "子": "戌", "丑": "戌", "寅": "丑", "卯": "丑", "辰": "丑", "巳": "辰", "午": "辰", "未": "辰", "申": "未", "酉": "未", "戌": "未", "亥": "戌" },
	"大耗": { "子": "巳未", "丑": "午申", "寅": "未酉", "卯": "申戌", "辰": "酉亥", "巳": "戌子", "午": "亥丑", "未": "子寅", "申": "丑卯", "酉": "寅辰", "戌": "卯巳", "亥": "辰午" }
};
var BAZI_MONTH_SHENS = {
	"天德": { "子": "巳", "丑": "庚", "寅": "丁", "卯": "申", "辰": "壬", "巳": "辛", "午": "亥", "未": "甲", "申": "癸", "酉": "寅", "戌": "丙", "亥": "乙" },
	"月德": { "子": "壬", "丑": "庚", "寅": "丙", "卯": "甲", "辰": "壬", "巳": "庚", "午": "丙", "未": "甲", "申": "壬", "酉": "庚", "戌": "丙", "亥": "甲" }
};
var BAZI_DAY_SHENS = {
	"将星": { "子": "子", "丑": "酉", "寅": "午", "卯": "卯", "辰": "子", "巳": "酉", "午": "午", "未": "卯", "申": "子", "酉": "酉", "戌": "午", "亥": "卯" },
	"华盖": { "子": "辰", "丑": "丑", "寅": "戌", "卯": "未", "辰": "辰", "巳": "丑", "午": "戌", "未": "未", "申": "辰", "酉": "丑", "戌": "戌", "亥": "未" },
	"驿马": { "子": "寅", "丑": "亥", "寅": "申", "卯": "巳", "辰": "寅", "巳": "亥", "午": "申", "未": "巳", "申": "寅", "酉": "亥", "戌": "申", "亥": "巳" },
	"劫煞": { "子": "巳", "丑": "寅", "寅": "亥", "卯": "申", "辰": "巳", "巳": "寅", "午": "亥", "未": "申", "申": "巳", "酉": "寅", "戌": "亥", "亥": "申" },
	"亡神": { "子": "亥", "丑": "申", "寅": "巳", "卯": "寅", "辰": "亥", "巳": "申", "午": "巳", "未": "寅", "申": "亥", "酉": "申", "戌": "巳", "亥": "寅" },
	"桃花": { "子": "酉", "丑": "午", "寅": "卯", "卯": "子", "辰": "酉", "巳": "午", "午": "卯", "未": "子", "申": "酉", "酉": "午", "戌": "卯", "亥": "子" }
};
var BAZI_G_SHENS = {
	"天乙": { "甲": "未丑", "乙": "申子", "丙": "酉亥", "丁": "酉亥", "戊": "未丑", "己": "申子", "庚": "未丑", "辛": "寅午", "壬": "卯巳", "癸": "卯巳" },
	"文昌": { "甲": "巳", "乙": "午", "丙": "申", "丁": "酉", "戊": "申", "己": "酉", "庚": "亥", "辛": "子", "壬": "寅", "癸": "丑" },
	"阳刃": { "甲": "卯", "乙": "", "丙": "午", "丁": "", "戊": "午", "己": "", "庚": "酉", "辛": "", "壬": "子", "癸": "" },
	"红艳": { "甲": "午", "乙": "午", "丙": "寅", "丁": "未", "戊": "辰", "己": "辰", "庚": "戌", "辛": "酉", "壬": "子", "癸": "申" }
};
var BAZI_SHENS_INFOS = {
	"孤辰": "孤僻、独处倾向偏强，月柱出现时更明显。",
	"寡宿": "类似孤辰，偏独立，情感与婚姻体验常较迟。",
	"大耗": "多与财物损耗、意外支出相关，单独出现不必过度放大。",
	"天德": "先天福泽类神煞，常主逢凶化缓。",
	"月德": "与天德相近，偏向贵人、缓冲和助力。",
	"将星": "主主见、气度、执行力。",
	"华盖": "常见于艺术、宗教、独处、审美与精神性。",
	"驿马": "多动、多迁移、多变动，也可指外出发展。",
	"劫煞": "与冲动、风险、竞争相关，需结合全局看。",
	"亡神": "多与心神耗散、暗耗、纠结有关。",
	"桃花": "多主人缘、审美、情感事件，也可能带来情感扰动。",
	"天乙": "贵人助力，常作解难之象。",
	"文昌": "偏学习、表达、文书、理解力。",
	"阳刃": "刚烈、决断、冲劲强，过旺时需制衡。",
	"红艳": "情感投入度高，审美与吸引力会更显。"
};
var BAZI_XINGXIU = {
	0: "角", 1: "亢", 2: "氐", 3: "房", 4: "心", 5: "尾", 6: "箕", 7: "斗",
	8: "牛", 9: "女", 10: "虚", 11: "危", 12: "室", 13: "壁", 14: "奎", 15: "娄",
	16: "胃", 17: "昴", 18: "毕", 19: "觜", 20: "参", 21: "井", 22: "鬼", 23: "柳",
	24: "星", 25: "张", 26: "翼", 27: "轸"
};
var BAZI_JIANCHU = {
	0: ["建", "气专而强，宜赴任、祈福、求财、交涉。"],
	1: ["除", "适合除旧布新、祭祀、祈福、入宅。"],
	2: ["满", "偏丰收圆满，宜嫁娶、祭祀、求财。"],
	3: ["平", "平常日，宜修补粉刷，不宜远行。"],
	4: ["定", "偏稳定成事，宜祈福、嫁娶、纳财。"],
	5: ["执", "偏守成，宜纳采、嫁娶、动土。"],
	6: ["破", "多破耗，不宜办大事。"],
	7: ["危", "宜入殓安葬，不宜嫁娶入宅。"],
	8: ["成", "成事日，宜开市、动土、安床、交易。"],
	9: ["收", "偏收获，宜祈福、嫁娶、交易。"],
	10: ["开", "适合开展新事务，宜开市、入宅、出行。"],
	11: ["闭", "偏闭藏，宜安门、伐木、修造。"]
};
var BAZI_GAN_HE_LABELS = {
	"甲己": "甲己合化土",
	"乙庚": "乙庚合化金",
	"丙辛": "丙辛合化水",
	"丁壬": "丁壬合化木",
	"戊癸": "戊癸合化火"
};
var BAZI_GAN_CHONG_LABELS = {
	"甲庚": "甲庚冲",
	"乙辛": "乙辛冲",
	"丙壬": "丙壬冲",
	"丁癸": "丁癸冲"
};
var BAZI_ZHI_ATTS = {
	"子": { "冲": "午", "刑": "卯", "被刑": "卯", "害": "未", "破": "酉", "六": "丑", "暗": "" },
	"丑": { "冲": "未", "刑": "戌", "被刑": "未", "害": "午", "破": "辰", "六": "子", "暗": "寅" },
	"寅": { "冲": "申", "刑": "巳", "被刑": "申", "害": "巳", "破": "亥", "六": "亥", "暗": "丑" },
	"卯": { "冲": "酉", "刑": "子", "被刑": "子", "害": "辰", "破": "午", "六": "戌", "暗": "申" },
	"辰": { "冲": "戌", "刑": "辰", "被刑": "辰", "害": "卯", "破": "丑", "六": "酉", "暗": "" },
	"巳": { "冲": "亥", "刑": "申", "被刑": "寅", "害": "寅", "破": "申", "六": "申", "暗": "" },
	"午": { "冲": "子", "刑": "午", "被刑": "午", "害": "丑", "破": "卯", "六": "未", "暗": "亥" },
	"未": { "冲": "丑", "刑": "丑", "被刑": "戌", "害": "子", "破": "戌", "六": "午", "暗": "" },
	"申": { "冲": "寅", "刑": "寅", "被刑": "巳", "害": "亥", "破": "巳", "六": "巳", "暗": "卯" },
	"酉": { "冲": "卯", "刑": "酉", "被刑": "酉", "害": "戌", "破": "子", "六": "辰", "暗": "" },
	"戌": { "冲": "辰", "刑": "未", "被刑": "丑", "害": "酉", "破": "未", "六": "卯", "暗": "" },
	"亥": { "冲": "巳", "刑": "亥", "被刑": "亥", "害": "申", "破": "寅", "六": "寅", "暗": "午" }
};
var BAZI_ZHI_SANHE = {
	"申子辰": "水局",
	"巳酉丑": "金局",
	"寅午戌": "火局",
	"亥卯未": "木局"
};
var BAZI_ZHI_SANHUI = {
	"亥子丑": "水局",
	"寅卯辰": "木局",
	"巳午未": "火局",
	"申酉戌": "金局"
};
var BAZI_GONG_HE = {
	"申辰": "子", "辰申": "子",
	"巳丑": "酉", "丑巳": "酉",
	"寅戌": "午", "戌寅": "午",
	"亥未": "卯", "未亥": "卯"
};
var BAZI_GONG_HUI = {
	"亥丑": "子", "丑亥": "子",
	"寅辰": "卯", "辰寅": "卯",
	"巳未": "午", "未巳": "午",
	"申戌": "酉", "戌申": "酉"
};
var BAZI_BRANCH_RELATION_NOTES = {
	"六合": "多主牵引、合作、彼此靠近，是否成助力还要看全局。",
	"六冲": "多主变化、拉扯、对立、迁动，常带明显波动感。",
	"相刑": "多主内耗、别扭、压力和反复，严重与否要看是否叠加。",
	"相害": "多主暗损、误解、牵制，容易表面平静而内里不顺。",
	"相破": "多主破耗、漏洞、松动，常见于关系或事务上的损耗。",
	"暗合": "多主隐性牵连、暗中呼应，力量通常弱于明合。",
	"三合局": "三支成局后，五行取向会更集中，往往放大某一类力量。",
	"三会局": "同季三支会聚，气势更整齐，环境倾向会更明显。"
};
var BAZI_ZHI_XING_DETAILS = {
	"寅巳": "寅巳申多作无恩之刑，容易急、硬、带冲突感。",
	"巳申": "寅巳申多作无恩之刑，容易急、硬、带冲突感。",
	"申寅": "寅巳申多作无恩之刑，容易急、硬、带冲突感。",
	"未丑": "丑未戌多作持势之刑，常见僵持、较劲、压力互顶。",
	"丑戌": "丑未戌多作持势之刑，常见僵持、较劲、压力互顶。",
	"戌未": "丑未戌多作持势之刑，常见僵持、较劲、压力互顶。",
	"子卯": "子卯多作无礼之刑，容易失衡、失和、情绪对冲。",
	"卯子": "子卯多作无礼之刑，容易失衡、失和、情绪对冲。"
};
var BAZI_ZHI_HAI_DETAILS = {
	"子未": "未害子，多主暗中牵制，不利骨肉与稳定感。",
	"午丑": "午害丑，多主火金相凌，事情上常见硬碰与损耗。",
	"寅巳": "寅巳相害，多主争进、较劲、合作中不易完全信任。",
	"卯辰": "卯害辰，多主以下凌上或节奏错位，久了易成压力。",
	"申亥": "申亥相害，多主嫉能、猜度、暗耗，表面和内里不总一致。",
	"酉戌": "酉戌相害，多主嫉妒与不容，容易在细节上起别扭。"
};

function baziEscapeHtml(text) {
	return String(text)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

function baziStripHtml(text) {
	return String(text || "").replace(/<[^>]*>/g, "");
}

function baziPad(num) {
	num = parseInt(num, 10);
	return num < 10 ? "0" + num : String(num);
}

function baziGetStemIndex(stem) {
	return p.ctg.indexOf(stem);
}

function baziGetBranchIndex(branch) {
	return p.cdz.indexOf(branch);
}

function baziGetElementNameByStem(stem) {
	var index = baziGetStemIndex(stem);
	return index >= 0 ? BAZI_WUXING_NAMES[p.wxtg[index]] : "";
}

function baziGetElementNameByBranch(branch) {
	var index = baziGetBranchIndex(branch);
	return index >= 0 ? BAZI_WUXING_NAMES[p.wxdz[index]] : "";
}

function baziGetShiShenIndex(dayStem, targetStem) {
	var dayIndex = baziGetStemIndex(dayStem);
	var targetIndex = baziGetStemIndex(targetStem);
	if (dayIndex < 0 || targetIndex < 0) {
		return -1;
	}
	return p.dgs[targetIndex][dayIndex];
}

function baziGetShiShenShort(dayStem, targetStem) {
	var index = baziGetShiShenIndex(dayStem, targetStem);
	return index >= 0 ? BAZI_SHISHEN_SHORT[index] : "";
}

function baziGetShiShenFull(dayStem, targetStem) {
	var index = baziGetShiShenIndex(dayStem, targetStem);
	return index >= 0 ? BAZI_SHISHEN_FULL[index] : "";
}

function baziGetMainHiddenStem(branch) {
	var list = BAZI_ZHI_HIDDEN_STEMS[branch] || [];
	return list.length ? list[0].gan : "";
}

function baziGetHiddenStemRows(dayStem, branch) {
	var list = BAZI_ZHI_HIDDEN_STEMS[branch] || [];
	var rows = [];
	for (var i = 0; i < list.length; i++) {
		rows.push({
			gan: list[i].gan,
			weight: list[i].weight,
			element: baziGetElementNameByStem(list[i].gan),
			shishen: baziGetShiShenShort(dayStem, list[i].gan)
		});
	}
	return rows;
}

function baziGetGzIndex(stem, branch) {
	return p.GZ(baziGetStemIndex(stem), baziGetBranchIndex(branch));
}

function baziGetNaYin(stem, branch) {
	var index = baziGetGzIndex(stem, branch);
	if (index < 0) {
		return "";
	}
	return BAZI_NAYIN[Math.floor(index / 2)] || "";
}

function baziGetEmptyBranches(dayStem, dayBranch) {
	var gzIndex = baziGetGzIndex(dayStem, dayBranch);
	if (gzIndex < 0) {
		return [];
	}
	return BAZI_EMPTY_BY_XUN[Math.floor(gzIndex / 10)] || [];
}

function baziGetStorageBranch(dayStem) {
	var element = baziGetElementNameByStem(dayStem);
	if (element === "金") return "丑";
	if (element === "木") return "未";
	if (element === "水") return "辰";
	if (element === "火") return "戌";
	if (element === "土") return "辰";
	return "";
}

function baziGetChangShengIndex(dayStem, branch) {
	var stemIndex = baziGetStemIndex(dayStem);
	var branchIndex = baziGetBranchIndex(branch);
	if (stemIndex < 0 || branchIndex < 0) {
		return -1;
	}
	return (24 + BAZI_DAY_STEM_STAGE_START[stemIndex] + Math.pow(-1, stemIndex) * branchIndex) % 12;
}

function baziGetChangShengShort(dayStem, branch) {
	var index = baziGetChangShengIndex(dayStem, branch);
	return index >= 0 ? BAZI_CHANGSHENG_SHORT[index] : "";
}

function baziGetChangShengFull(dayStem, branch) {
	var index = baziGetChangShengIndex(dayStem, branch);
	return index >= 0 ? p.czs[index] : "";
}

function baziIncludesChar(source, target) {
	return source && source.indexOf(target) !== -1;
}

function baziUnique(list) {
	var seen = {};
	var result = [];
	for (var i = 0; i < list.length; i++) {
		if (!list[i] || seen[list[i]]) {
			continue;
		}
		seen[list[i]] = true;
		result.push(list[i]);
	}
	return result;
}

function baziHasAllChars(sourceSet, chars) {
	for (var i = 0; i < chars.length; i++) {
		if (!sourceSet[chars.charAt(i)]) {
			return false;
		}
	}
	return true;
}

function baziCollectStemRelations(gans) {
	var names = ["年", "月", "日", "时"];
	var result = [];
	for (var i = 0; i < gans.length; i++) {
		for (var j = i + 1; j < gans.length; j++) {
			var pair = gans[i] + gans[j];
			var reverse = gans[j] + gans[i];
			var he = BAZI_GAN_HE_LABELS[pair] || BAZI_GAN_HE_LABELS[reverse];
			var chong = BAZI_GAN_CHONG_LABELS[pair] || BAZI_GAN_CHONG_LABELS[reverse];
			if (he) {
				result.push(names[i] + "干与" + names[j] + "干：" + he);
			}
			if (chong) {
				result.push(names[i] + "干与" + names[j] + "干：" + chong);
			}
		}
	}
	return result;
}

function baziCollectBranchPairRelations(zhis) {
	var names = ["年", "月", "日", "时"];
	var result = [];
	var notes = [];
	var seenNotes = {};
	for (var i = 0; i < zhis.length; i++) {
		for (var j = i + 1; j < zhis.length; j++) {
			var left = zhis[i];
			var right = zhis[j];
			var leftAtt = BAZI_ZHI_ATTS[left] || {};
			var rightAtt = BAZI_ZHI_ATTS[right] || {};
			var labels = [];
			var pairKey = left + right;
			var reverseKey = right + left;
			if (leftAtt["六"] === right || rightAtt["六"] === left) labels.push("六合");
			if (leftAtt["冲"] === right || rightAtt["冲"] === left) labels.push("六冲");
			if (leftAtt["刑"] === right || rightAtt["刑"] === left || leftAtt["被刑"] === right || rightAtt["被刑"] === left) labels.push("相刑");
			if (leftAtt["害"] === right || rightAtt["害"] === left) labels.push("相害");
			if (leftAtt["破"] === right || rightAtt["破"] === left) labels.push("相破");
			if (leftAtt["暗"] === right || rightAtt["暗"] === left) labels.push("暗合");
			if (labels.length) {
				result.push(names[i] + "支" + left + " 与 " + names[j] + "支" + right + "：" + labels.join("、"));
				for (var k = 0; k < labels.length; k++) {
					var label = labels[k];
					var noteKey = label;
					var noteText = BAZI_BRANCH_RELATION_NOTES[label] || "";
					if (label === "相刑") {
						noteText = BAZI_ZHI_XING_DETAILS[pairKey] || BAZI_ZHI_XING_DETAILS[reverseKey] || noteText;
					}
					if (label === "相害") {
						noteText = BAZI_ZHI_HAI_DETAILS[pairKey] || BAZI_ZHI_HAI_DETAILS[reverseKey] || noteText;
					}
					if (noteText && !seenNotes[noteKey + ":" + noteText]) {
						seenNotes[noteKey + ":" + noteText] = true;
						notes.push(label + "：" + noteText);
					}
				}
			}
		}
	}
	return {
		items: result,
		notes: notes
	};
}

function baziCollectBranchGroupRelations(zhis) {
	var setMap = {};
	var result = [];
	var notes = [];
	for (var i = 0; i < zhis.length; i++) {
		setMap[zhis[i]] = true;
	}
	for (var key in BAZI_ZHI_SANHE) {
		if (baziHasAllChars(setMap, key)) {
			result.push("三合局：" + key + "化" + BAZI_ZHI_SANHE[key]);
			notes.push("三合局：" + BAZI_BRANCH_RELATION_NOTES["三合局"]);
		}
	}
	for (var key2 in BAZI_ZHI_SANHUI) {
		if (baziHasAllChars(setMap, key2)) {
			result.push("三会局：" + key2 + "化" + BAZI_ZHI_SANHUI[key2]);
			notes.push("三会局：" + BAZI_BRANCH_RELATION_NOTES["三会局"]);
		}
	}
	return {
		items: result,
		notes: baziUnique(notes)
	};
}

function baziCollectGongAndJia(analysis) {
	var names = ["年", "月", "日", "时"];
	var items = [];
	var notes = [];
	var zhis = analysis.zhis || [];
	var gans = analysis.gans || [];

	for (var i = 0; i < zhis.length; i++) {
		for (var j = i + 1; j < zhis.length; j++) {
			var z1 = zhis[i];
			var z2 = zhis[j];
			var idx1 = baziGetBranchIndex(z1);
			var idx2 = baziGetBranchIndex(z2);
			if (idx1 < 0 || idx2 < 0) {
				continue;
			}
			if (gans[i] === gans[j]) {
				var diff = Math.abs(idx1 - idx2);
				if (diff === 2) {
					items.push(names[i] + "柱与" + names[j] + "柱同干" + gans[i] + "，地支" + z1 + z2 + "夹出" + p.cdz[(idx1 + idx2) / 2]);
				} else if (diff === 10) {
					items.push(names[i] + "柱与" + names[j] + "柱同干" + gans[i] + "，地支" + z1 + z2 + "夹出" + p.cdz[(idx1 + idx2) % 12]);
				}
			}

			var heKey = z1 + z2;
			var huiKey = z1 + z2;
			if (BAZI_GONG_HE[heKey] && zhis.indexOf(BAZI_GONG_HE[heKey]) === -1) {
				items.push(names[i] + "支" + z1 + " 与 " + names[j] + "支" + z2 + "：三合拱" + BAZI_GONG_HE[heKey]);
			}
			if (BAZI_GONG_HUI[huiKey] && zhis.indexOf(BAZI_GONG_HUI[huiKey]) === -1) {
				items.push(names[i] + "支" + z1 + " 与 " + names[j] + "支" + z2 + "：三会拱" + BAZI_GONG_HUI[huiKey]);
			}
		}
	}

	if (items.length) {
		notes.push("夹：多指同干两支把中间一支夹出，常作潜在气机参考。");
		notes.push("拱：多指两支拱出中神，常作辅佐信息，不单独定吉凶。");
	}

	return {
		items: items,
		notes: notes
	};
}

function baziCollectEmptyDetails(analysis) {
	var names = ["年", "月", "日", "时"];
	var items = [];
	var notes = [];
	var emptyPillars = [];
	var dayPillar = analysis.gans[2] + analysis.zhis[2];
	for (var i = 0; i < analysis.pillars.length; i++) {
		if (analysis.pillars[i].empty) {
			emptyPillars.push(names[i] + "柱");
		}
	}
	items.push("日柱旬空：" + dayPillar + " -> " + (analysis.emptyBranches.join("、") || "无"));
	if (emptyPillars.length) {
		items.push("本命落空：" + emptyPillars.join("、"));
	} else {
		items.push("本命落空：四柱本位未直接落入旬空");
	}
	notes.push("空亡：以日柱旬空为准，落空之支多主该处之力有悬、迟、虚、隔之感。");
	return {
		items: items,
		notes: notes
	};
}

function baziGetShens(gans, zhis, ganValue, zhiValue, dayStem) {
	var allShens = [];
	var item;
	for (item in BAZI_YEAR_SHENS) {
		if (baziIncludesChar(BAZI_YEAR_SHENS[item][zhis[0]], zhiValue)) {
			allShens.push(item);
		}
	}
	for (item in BAZI_MONTH_SHENS) {
		var monthRule = BAZI_MONTH_SHENS[item][zhis[1]] || "";
		if (baziIncludesChar(monthRule, ganValue) || baziIncludesChar(monthRule, zhiValue)) {
			allShens.push(item);
		}
	}
	for (item in BAZI_DAY_SHENS) {
		if (baziIncludesChar(BAZI_DAY_SHENS[item][zhis[2]], zhiValue)) {
			allShens.push(item);
		}
	}
	for (item in BAZI_G_SHENS) {
		if (baziIncludesChar(BAZI_G_SHENS[item][dayStem], zhiValue)) {
			allShens.push(item);
		}
	}
	return baziUnique(allShens);
}

function baziBuildCoreAnalysis(fm) {
	var gans = fm.ctg.slice(0);
	var zhis = fm.cdz.slice(0);
	var dayStem = gans[2];
	var emptyBranches = baziGetEmptyBranches(gans[2], zhis[2]);
	var storageBranch = baziGetStorageBranch(dayStem);
	var ganShens = [];
	var zhiShens = [];
	var zhiShenAll = [];
	var zhiShenStrings = [];
	var pillarDetails = [];
	var scores = { "金": 0, "木": 0, "水": 0, "火": 0, "土": 0 };
	var ganScores = { "甲": 0, "乙": 0, "丙": 0, "丁": 0, "戊": 0, "己": 0, "庚": 0, "辛": 0, "壬": 0, "癸": 0 };
	var supportScore = 0;
	var opposeScore = 0;
	var allShens = [];
	var i;

	for (i = 0; i < gans.length; i++) {
		ganShens[i] = i === 2 ? "--" : baziGetShiShenShort(dayStem, gans[i]);
	}

	for (i = 0; i < zhis.length; i++) {
		var hiddenRows = baziGetHiddenStemRows(dayStem, zhis[i]);
		var mainHidden = hiddenRows.length ? hiddenRows[0] : null;
		var hiddenText = [];
		var branchAllShens = [];
		var branchShens = baziGetShens(gans, zhis, gans[i], zhis[i], dayStem);
		allShens = allShens.concat(branchShens);

		if (mainHidden) {
			zhiShens[i] = mainHidden.shishen;
		} else {
			zhiShens[i] = "";
		}

		for (var j = 0; j < hiddenRows.length; j++) {
			hiddenText.push(hiddenRows[j].gan + hiddenRows[j].element + hiddenRows[j].shishen);
			branchAllShens.push(hiddenRows[j].shishen);
			zhiShenAll.push(hiddenRows[j].shishen);
		}
		zhiShenStrings[i] = branchAllShens.join("");

		pillarDetails[i] = {
			gan: gans[i],
			ganShen: ganShens[i],
			zhi: zhis[i],
			zhiShen: zhiShens[i],
			hidden: hiddenRows,
			hiddenText: hiddenText.join("　"),
			changSheng: baziGetChangShengShort(dayStem, zhis[i]),
			changShengFull: baziGetChangShengFull(dayStem, zhis[i]),
			nayin: baziGetNaYin(gans[i], zhis[i]),
			empty: emptyBranches.indexOf(zhis[i]) !== -1,
			shens: branchShens
		};
	}

	for (i = 0; i < gans.length; i++) {
		var element = baziGetElementNameByStem(gans[i]);
		scores[element] += 5;
		ganScores[gans[i]] += 5;
		var stemShen = i === 2 ? "比" : ganShens[i];
		if (stemShen === "比" || stemShen === "劫" || stemShen === "印" || stemShen === "枭") {
			supportScore += 5;
		} else {
			opposeScore += 5;
		}
	}

	for (i = 0; i < zhis.length + 1; i++) {
		var branch = zhis[i < 4 ? i : 1];
		var branchRows = BAZI_ZHI_HIDDEN_STEMS[branch] || [];
		for (var k = 0; k < branchRows.length; k++) {
			var branchElement = baziGetElementNameByStem(branchRows[k].gan);
			var branchShenName = baziGetShiShenShort(dayStem, branchRows[k].gan);
			scores[branchElement] += branchRows[k].weight;
			ganScores[branchRows[k].gan] += branchRows[k].weight;
			if (branchShenName === "比" || branchShenName === "劫" || branchShenName === "印" || branchShenName === "枭") {
				supportScore += branchRows[k].weight;
			} else {
				opposeScore += branchRows[k].weight;
			}
		}
	}

	var branchStatuses = [];
	var weak = true;
	var libraryCount = 0;
	for (i = 0; i < zhis.length; i++) {
		branchStatuses[i] = pillarDetails[i].changSheng;
		if (branchStatuses[i] === "长" || branchStatuses[i] === "帝" || branchStatuses[i] === "建") {
			weak = false;
		}
		if (zhis[i] === storageBranch) {
			libraryCount += 1;
		}
	}
	if (weak) {
		var biCount = 0;
		for (i = 0; i < ganShens.length; i++) {
			if (ganShens[i] === "比") {
				biCount += 1;
			}
		}
		if (biCount + libraryCount > 2) {
			weak = false;
		}
	}

	var stemRelations = baziCollectStemRelations(gans);
	var branchPairRelations = baziCollectBranchPairRelations(zhis);
	var branchGroupRelations = baziCollectBranchGroupRelations(zhis);
	var gongAndJia = baziCollectGongAndJia({
		gans: gans,
		zhis: zhis,
		pillars: pillarDetails
	});
	var emptyDetails = baziCollectEmptyDetails({
		gans: gans,
		zhis: zhis,
		pillars: pillarDetails,
		emptyBranches: emptyBranches
	});

	return {
		gans: gans,
		zhis: zhis,
		dayStem: dayStem,
		ganShens: ganShens,
		zhiShens: zhiShens,
		zhiShenStrings: zhiShenStrings,
		pillars: pillarDetails,
		emptyBranches: emptyBranches,
		storageBranch: storageBranch,
		scores: scores,
		ganScores: ganScores,
		supportScore: supportScore,
		opposeScore: opposeScore,
		weak: weak,
		branchStatuses: branchStatuses,
		allShens: baziUnique(allShens),
		stemRelations: stemRelations,
		branchPairRelations: branchPairRelations.items,
		branchGroupRelations: branchGroupRelations.items,
		relationNotes: baziUnique((branchPairRelations.notes || []).concat(branchGroupRelations.notes || []).concat(gongAndJia.notes || []).concat(emptyDetails.notes || [])),
		gongAndJia: gongAndJia.items,
		emptyDetails: emptyDetails.items,
		natalSummary: supportScore >= opposeScore ? "扶助力量略强" : "泄耗克制略强"
	};
}

function baziFormatLunar(nl) {
	if (!nl || !nl[4]) {
		return "";
	}
	return nl[0] + "年 " + (nl[3] ? "闰" : "") + nl[4].ym + p.dxd[nl[2] - 1];
}

function baziGetXingXiu(yy, mm, dd) {
	if (yy <= 0) {
		return "";
	}
	var base = p.Jdays(1, 1, 4, 12, 0, 0);
	var now = p.Jdays(yy, mm, dd, 12, 0, 0);
	if (base === false || now === false) {
		return "";
	}
	var idx = ((Math.floor(now - base) % 28) + 28) % 28;
	return BAZI_XINGXIU[idx] || "";
}

function baziGetJianChu(monthBranch, dayBranch) {
	var monthIndex = baziGetBranchIndex(monthBranch);
	var dayIndex = baziGetBranchIndex(dayBranch);
	if (monthIndex < 0 || dayIndex < 0) {
		return null;
	}
	var seq = 12 - monthIndex;
	return BAZI_JIANCHU[(dayIndex + seq) % 12] || null;
}

function baziGetSummaryKey(analysis) {
	if (!analysis || !analysis.dayStem || !analysis.gans || !analysis.zhis || analysis.gans.length < 4 || analysis.zhis.length < 4) {
		return "";
	}
	return analysis.dayStem + "日" + analysis.gans[3] + analysis.zhis[3];
}

function baziGetMonthKey(analysis) {
	if (!analysis || !analysis.dayStem || !analysis.zhis || analysis.zhis.length < 2) {
		return "";
	}
	return analysis.dayStem + analysis.zhis[1];
}

function baziRenderTextCard(title, keyText, content) {
	var html = '';
	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">' + baziEscapeHtml(title) + '</div>';
	if (!content) {
		if (keyText) {
			html += '<div style="margin-bottom:6px;color:#666;">匹配键：' + baziEscapeHtml(keyText) + '</div>';
		}
		html += '<div style="color:#666;line-height:1.8;">暂未收录</div>';
		html += '</div>';
		return html;
	}
	html += '<details>';
	html += '<summary style="cursor:pointer;line-height:1.7;outline:none;">';
	if (keyText) {
		html += '匹配键：' + baziEscapeHtml(keyText);
	} else {
		html += '展开参考';
	}
	html += '</summary>';
	html += '<div style="margin-top:8px;white-space:pre-wrap;line-height:1.8;">' + baziEscapeHtml(content) + '</div>';
	html += '</details>';
	html += '</div>';
	return html;
}

function baziRenderClassicTexts(analysis) {
	var summaries = window.BAZI_SUMMARYS || {};
	var months = window.BAZI_MONTHS || {};
	var summaryKey = baziGetSummaryKey(analysis);
	var monthKey = baziGetMonthKey(analysis);
	var summaryText = summaries[summaryKey] || "";
	var monthText = months[monthKey] || "";
	var html = '';

	html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;align-items:start;">';
	html += baziRenderTextCard("时柱命例", summaryKey, summaryText);
	html += baziRenderTextCard("月令参考", monthKey, monthText);
	html += '</div>';

	return html;
}

function baziRenderPillarTable(analysis) {
	var titles = ["年柱", "月柱", "日柱", "时柱"];
	var html = '';
	html += '<div style="overflow-x:auto;">';
	html += '<table style="width:100%;border-collapse:collapse;text-align:left;table-layout:fixed;">';
	html += '<colgroup>';
	html += '<col style="width:108px;">';
	html += '<col style="width:calc((100% - 108px) / 4);">';
	html += '<col style="width:calc((100% - 108px) / 4);">';
	html += '<col style="width:calc((100% - 108px) / 4);">';
	html += '<col style="width:calc((100% - 108px) / 4);">';
	html += '</colgroup>';
	html += '<thead><tr>';
	html += '<th style="padding:6px 8px;border-bottom:1px solid #ddd;">项目</th>';
	for (var i = 0; i < 4; i++) {
		html += '<th style="padding:6px 8px;border-bottom:1px solid #ddd;">' + titles[i] + '</th>';
	}
	html += '</tr></thead><tbody>';

	var rows = [
		["天干", function (p) { return p.gan; }],
		["天干十神", function (p) { return p.ganShen || "--"; }],
		["地支", function (p) { return p.zhi; }],
		["地支主气", function (p) { return p.zhiShen || ""; }],
		["十二长生", function (p) { return p.changSheng + " " + p.changShengFull; }],
		["藏干", function (p) { return p.hiddenText || "—"; }],
		["纳音", function (p) { return p.nayin || "—"; }],
		["空亡", function (p) { return p.empty ? "空" : "—"; }],
		["神煞", function (p) { return p.shens.length ? p.shens.join("、") : "—"; }]
	];

	for (var r = 0; r < rows.length; r++) {
		html += '<tr>';
		html += '<td style="padding:6px 8px;border-bottom:1px solid #f0f0f0;color:#666;white-space:nowrap;">' + rows[r][0] + '</td>';
		for (var c = 0; c < 4; c++) {
			html += '<td style="padding:6px 8px;border-bottom:1px solid #f0f0f0;vertical-align:top;word-break:break-word;">' + baziEscapeHtml(rows[r][1](analysis.pillars[c])) + '</td>';
		}
		html += '</tr>';
	}

	html += '</tbody></table>';
	html += '</div>';
	return html;
}

function baziRenderScoreTable(analysis) {
	var wuxingOrder = ["金", "木", "水", "火", "土"];
	var ganOrder = p.ctg;
	var html = '';

	html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;">';

	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">五行分数</div>';
	for (var i = 0; i < wuxingOrder.length; i++) {
		var name = wuxingOrder[i];
		html += '<div style="margin:2px 0;">' + name + '：' + analysis.scores[name] + '</div>';
	}
	html += '</div>';

	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">扶抑概览</div>';
	html += '<div style="margin:2px 0;">日主：' + analysis.dayStem + '（' + baziGetElementNameByStem(analysis.dayStem) + '）</div>';
	html += '<div style="margin:2px 0;">扶助：' + analysis.supportScore + '</div>';
	html += '<div style="margin:2px 0;">泄耗克制：' + analysis.opposeScore + '</div>';
	html += '<div style="margin:2px 0;">强弱：' + (analysis.weak ? "偏弱" : "偏强/不弱") + '</div>';
	html += '<div style="margin:2px 0;">判断：' + analysis.natalSummary + '</div>';
	html += '</div>';

	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">天干分数</div>';
	for (var j = 0; j < ganOrder.length; j++) {
		html += '<div style="margin:2px 0;">' + ganOrder[j] + '：' + analysis.ganScores[ganOrder[j]] + '</div>';
	}
	html += '</div>';

	html += '</div>';
	return html;
}

function baziRenderGods(analysis) {
	if (!analysis.allShens.length) {
		return '<div>本命未检出明显神煞。</div>';
	}
	var html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;">';
	for (var i = 0; i < analysis.allShens.length; i++) {
		var name = analysis.allShens[i];
		html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
		html += '<div style="font-weight:bold;margin-bottom:4px;">' + name + '</div>';
		html += '<div style="color:#555;line-height:1.6;">' + baziEscapeHtml(BAZI_SHENS_INFOS[name] || "可作为辅助参考，不宜单独定论。") + '</div>';
		html += '</div>';
	}
	html += '</div>';
	return html;
}

function baziRenderRelations(analysis) {
	var sections = [
		{ title: "天干关系", items: analysis.stemRelations || [] },
		{ title: "地支关系", items: (analysis.branchPairRelations || []).concat(analysis.branchGroupRelations || []) },
		{ title: "拱夹与空亡", items: (analysis.gongAndJia || []).concat(analysis.emptyDetails || []) },
		{ title: "关系说明", items: analysis.relationNotes || [] }
	];
	var html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;">';

	for (var i = 0; i < sections.length; i++) {
		var section = sections[i];
		html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
		html += '<div style="font-weight:bold;margin-bottom:6px;">' + baziEscapeHtml(section.title) + '</div>';
		if (!section.items.length) {
			html += '<div style="color:#666;">未见明显关系</div>';
		} else {
			for (var j = 0; j < section.items.length; j++) {
				html += '<div style="margin:4px 0;line-height:1.7;">' + baziEscapeHtml(section.items[j]) + '</div>';
			}
		}
		html += '</div>';
	}

	html += '</div>';
	return html;
}

function baziRenderDayun(fm, analysis) {
	var html = '';
	html += '<div style="margin-bottom:8px;">' + baziEscapeHtml(fm.qyy_desc) + '，' + baziEscapeHtml(fm.qyy_desc2) + '</div>';

	for (var i = 0; i < fm.dy.length; i++) {
		var item = fm.dy[i];
		var gan = item.zfma;
		var zhi = item.zfmb;
		var branchShen = baziGetShiShenShort(analysis.dayStem, baziGetMainHiddenStem(zhi));
		var shens = baziGetShens(analysis.gans, analysis.zhis, gan, zhi, analysis.dayStem);
		var empty = analysis.emptyBranches.indexOf(zhi) !== -1 ? " 空亡" : "";
		var summary = item.zqage + "~" + item.zboz + "岁 [" + item.syear + "-" + item.eyear + "年] " + gan + zhi +
			" 干神:" + baziGetShiShenShort(analysis.dayStem, gan) +
			" 支神:" + branchShen +
			" " + item.nzsc + empty;

		html += '<details style="margin:8px 0;padding:8px 10px;border:1px solid #eee;border-radius:8px;">';
		html += '<summary style="cursor:pointer;line-height:1.7;">' + baziEscapeHtml(summary) + (shens.length ? ' 神煞:' + baziEscapeHtml(shens.join("、")) : '') + '</summary>';
		html += '<div style="margin-top:8px;line-height:1.75;">';
		html += '<div><b>十神：</b>' + baziEscapeHtml(baziGetShiShenFull(analysis.dayStem, gan)) + ' / ' + baziEscapeHtml(branchShen || "—") + '</div>';
		html += '<div><b>藏干：</b>' + baziEscapeHtml((baziGetHiddenStemRows(analysis.dayStem, zhi).map(function (row) {
			return row.gan + row.element + row.shishen;
		}).join("　")) || "—") + '</div>';
		html += '<div><b>纳音：</b>' + baziEscapeHtml(baziGetNaYin(gan, zhi)) + '</div>';
		html += '<div><b>神煞：</b>' + (shens.length ? baziEscapeHtml(shens.join("、")) : "—") + '</div>';

		if (item.ly && item.ly.length) {
			html += '<div style="margin-top:8px;"><b>流年：</b></div>';
			html += '<div style="margin-top:4px;">';
			for (var j = 0; j < item.ly.length; j++) {
				var ly = item.ly[j];
				var lyGan = ly.lye.charAt(0);
				var lyZhi = ly.lye.charAt(1);
				var lyBranchShen = baziGetShiShenShort(analysis.dayStem, baziGetMainHiddenStem(lyZhi));
				var lyShens = baziGetShens(analysis.gans, analysis.zhis, lyGan, lyZhi, analysis.dayStem);
				html += '<div style="padding:4px 0;border-bottom:' + (j === item.ly.length - 1 ? '0' : '1px dashed #f0f0f0') + ';">';
				html += baziEscapeHtml(ly.age + "岁 " + ly.year + "年 " + ly.lye);
				html += '　干神:' + baziEscapeHtml(baziGetShiShenShort(analysis.dayStem, lyGan));
				html += '　支神:' + baziEscapeHtml(lyBranchShen || "—");
				if (analysis.emptyBranches.indexOf(lyZhi) !== -1) {
					html += '　空亡';
				}
				if (lyShens.length) {
					html += '　神煞:' + baziEscapeHtml(lyShens.join("、"));
				}
				html += '</div>';
			}
			html += '</div>';
		}

		html += '</div></details>';
	}

	return html;
}

function baziRenderOverview(ob, fm, analysis, jd, inputDate) {
	var xingxiu = baziGetXingXiu(inputDate.yy, inputDate.mm, inputDate.dd);
	var jianchu = baziGetJianChu(analysis.zhis[1], analysis.zhis[2]);
	var cleanTimeSequence = baziStripHtml(ob.bz_JS).replace(/\s+/g, " ").trim();
	var html = '';

	html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;align-items:start;">';

	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">基础信息</div>';
	html += '<div>公历：' + baziEscapeHtml(inputDate.yy + "-" + baziPad(inputDate.mm) + "-" + baziPad(inputDate.dd) + " " + inputDate.time) + '</div>';
	html += '<div>农历：' + baziEscapeHtml(baziFormatLunar(fm.nl)) + '</div>';
	html += '<div>命造：' + baziEscapeHtml(fm.mz + "造（" + fm.xb + "）") + '</div>';
	html += '<div>生肖 / 星座：' + baziEscapeHtml(fm.sx + " / " + fm.xz) + '</div>';
	html += '<div>日主：' + baziEscapeHtml(analysis.dayStem + "（" + baziGetElementNameByStem(analysis.dayStem) + "）") + '</div>';
	html += '<div>旬空：' + baziEscapeHtml(analysis.emptyBranches.join("、") || "无") + '</div>';
	html += '</div>';

	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">时间与历法</div>';
	html += '<div>儒略日：' + int2(jd + 0.5) + '</div>';
	html += '<div>距 J2000：' + int2(jd + 0.5 - J2000) + ' 日</div>';
	html += '<div>真太阳时：' + baziEscapeHtml(ob.bz_zty) + '</div>';
	html += '<div>平太阳时：' + baziEscapeHtml(ob.bz_pty) + '</div>';
	html += '<div>纪时：' + baziEscapeHtml(cleanTimeSequence) + '</div>';
	html += '</div>';

	html += '<div style="padding:10px;border:1px solid #eee;border-radius:8px;grid-column:1 / -1;">';
	html += '<div style="font-weight:bold;margin-bottom:6px;">补充参考</div>';
	html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 12px;">';
	html += '<div>阴阳：阳 ' + fm.nyy[0] + ' / 阴 ' + fm.nyy[1] + '</div>';
	html += '<div>地支十二长生：' + baziEscapeHtml(analysis.branchStatuses.join("、")) + '</div>';
	html += '<div>日主库位：' + baziEscapeHtml(analysis.storageBranch || "—") + '</div>';
	html += '<div>星宿：' + baziEscapeHtml(xingxiu || "暂不显示") + '</div>';
	html += '<div style="grid-column:1 / -1;">建除：' + (jianchu ? baziEscapeHtml(jianchu[0] + " - " + jianchu[1]) : "暂不显示") + '</div>';
	html += '</div>';
	html += '</div>';

	html += '</div>';
	return html;
}

function baziBuildHtml(ob, fm, analysis, jd, inputDate) {
	var html = '';

	html += '<div style="line-height:1.75;max-width:980px;margin:0 auto;">';
	html += '<div style="margin-bottom:12px;"><font color=red><b>[八字]：</b></font>' +
		baziEscapeHtml(ob.bz_jn + "年 " + ob.bz_jy + "月 " + ob.bz_jr + "日 " + ob.bz_js + "时") +
		'</div>';

	html += baziRenderOverview(ob, fm, analysis, jd, inputDate);

	html += '<div style="margin:14px 0 8px;font-weight:bold;color:#a00;">[四柱结构]</div>';
	html += baziRenderPillarTable(analysis);

	html += '<div style="margin:14px 0 8px;font-weight:bold;color:#a00;">[干支关系]</div>';
	html += baziRenderRelations(analysis);

	html += '<div style="margin:14px 0 8px;font-weight:bold;color:#a00;">[五行与强弱]</div>';
	html += baziRenderScoreTable(analysis);

	html += '<div style="margin:14px 0 8px;font-weight:bold;color:#a00;">[神煞参考]</div>';
	html += baziRenderGods(analysis);

	html += '<div style="margin:14px 0 8px;font-weight:bold;color:#a00;">[古籍参考]</div>';
	html += baziRenderClassicTexts(analysis);

	html += '<div style="margin:14px 0 8px;font-weight:bold;color:#a00;">[大运]</div>';
	html += baziRenderDayun(fm, analysis);

	html += '</div>';
	return html;
}

function ML_calc() {
	var ob = {};
	var t = timeStr2hour(Cml_his.value);
	var secTo = t * 3600;
	var jd = JD.JD(year2Ayear(Cml_y.value), Cml_m.value - 0, Cml_d.value - 0 + t / 24);
	var yy = parseInt(Cml_y.value, 10);
	var mm = parseInt(Cml_m.value, 10);
	var dd = parseInt(Cml_d.value, 10);
	var his = Cml_his.value.split(":");
	var hh = parseInt(his[0], 10) || 0;
	var mt = parseInt(his[1], 10) || 0;
	var ss = parseInt(his[2], 10) || 0;
	var J = parseFloat(Cp11_J.value || "116.443");
	var W = 39.9214;
	var genderSelect = document.getElementById("Cml_xb");
	var xb = genderSelect ? parseInt(genderSelect.value, 10) || 0 : 0;

	ob.bz_pty = bdptys(secTo, Cp11_J.value);
	obb.mingLiBaZi(jd + curTZ / 24 - J2000, Cp11_J.value / radd, ob);

	var fm = p.fatemaps(xb, yy, mm, dd, hh, mt, ss, J, W);
	if (!fm) {
		Cal6.innerHTML = '<font color=red>输入日期无效，无法排盘。</font>';
		return;
	}

	var analysis = baziBuildCoreAnalysis(fm);
	Cal6.innerHTML = baziBuildHtml(ob, fm, analysis, jd, {
		yy: yy,
		mm: mm,
		dd: dd,
		time: Cml_his.value
	});
}

function ML_settime() {
	set_date_screen(1);
	ML_calc();
}
