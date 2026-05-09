const partList = {
    list: [{
            cnName: '穴位',
            mat: 'pifu',
            ava: 'xuewei',
            objName: 'M_001Acupoint',
            role: 'xw',
        },

        {
            cnName: '肌肉',
            mat: 'muscle',
            ava: 'jirou',
            objName: 'Muscles_system',
            role: 'jr',
        },
        {
            cnName: '起止点',
            mat: 'danse',
            ava: 'fuzhuodian',
            objName: 'Muscle_origin_and_insertion',
            role: 'lb',
        },
        {
            cnName: '骨连接',
            ava: 'gulianjie',
            mat: 'gulianjie',
            objName: '004Aticulation',
            role: 'glj',
        },
        {
            cnName: '骨骼',
            mat: 'bone',
            ava: 'guge',
            objName: 'Bones_system',
            role: 'gg',
        },

        {
            cnName: '静脉',
            mat: 'danse',
            ava: 'jingmai',
            objName: '006Systemic circulation venous',
            role: 'jm',
        },
        {
            cnName: '动脉',
            mat: 'danse',
            ava: 'dongmai',
            objName: '007Systemic circulation arterial',
            role: 'dm',
        },

        {
            cnName: '心脏',
            mat: 'xinzang',
            ava: 'xin',
            objName: '008Heart system',
            role: 'xz',
        },
        {
            cnName: '中枢',
            mat: 'zhongshu',
            ava: 'naobu',
            objName: '009Central',
            role: 'zs',
        },
        {
            cnName: '周围',
            mat: 'zhongshu',
            ava: 'shenjing',
            objName: '010Peripheral',
            role: 'zwsj',
        },
        {
            cnName: '淋巴',
            mat: 'danse',
            ava: 'linba',
            objName: '011Lymphatic system',
            role: 'lb',
        },
        {
            cnName: '器官',
            mat: 'qiguan',
            ava: 'qiguan',
            objName: '012Organ',
            role: 'qg',
        },
        {
            cnName: '眼',
            mat: 'zhongshu',
            ava: 'yanjing',
            objName: 'eyeObj',
            role: 'pf',
        },
        {
            cnName: '耳',
            mat: 'muscle',
            ava: 'erduo',
            objName: 'earObj',
            role: 'pf',
        }
    ]
}

function parseURL(url) {
    var a = document.createElement('a');
    a.href = url;
    return {
        params: (function () {
            var ret = {},
                seg = a.search.replace(/^\?/, '').split('&'),
                len = seg.length,
                i = 0,
                s;
            for (; i < len; i++) {
                if (!seg[i]) {
                    continue;
                }
                s = seg[i].split('=');
                ret[s[0]] = s[1];
            }
            return ret;
        })()
    }
}

const urlPart = parseURL(window.location.href).params.part||'hand'
const tokenurl = parseURL(window.location.href).params.token
const TOKEN = tokenurl?'Bearer '+tokenurl:null
const netPre = 'https://www.3dbody.com'

const materialArr = {

    bone: {
        normalScale:1,
        roughness: .4,
        originOrder:6,
    },
    gulianjie: {
        normalScale:.5,
        roughness: .45,
        originOrder:18,
    },
    pifu: {
        normalScale:1,
        roughness: .6,
        originOrder:20,
    },
    muscle: {
        normalScale:1.4,
        roughness: .4,
        originOrder:18,
    },
    rendai: {
        normalScale: .01,
        roughness: .37,
        originOrder:18,
    },
    naobu: {
        normalScale: .5,
        roughness: .37,
        originOrder:2,
    },
    danse: {
        normalScale: 0.3,
        roughness: .35,
        env:true
    },
    zhongshu: {
        normalScale: 1.5,
        roughness: .37,
    },
    xinzang:{
        normalScale: 1.4,
        roughness: .4,
    },
    qiguan: {
        normalScale: 1,
        roughness: .35,
        originOrder:4,
    }
}




