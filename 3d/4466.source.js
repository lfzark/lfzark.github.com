module.exports = function (e, t, n) {
  "use strict";
  (n.r(t), n.d(t, { default: () => s }));
  var r = n(1390),
    o = n(593),
    i = n(1299);
  (n.n(i)()(THREE), n(9461)(THREE));
  n(3122)(THREE);
  function a(e) {
    const t = String(e).replace(/^\/+/, ""),
      n = ("undefined" !== typeof process ? String(".") : "").replace(
        /\/+$/,
        "",
      );
    return "" === n || "/" === n
      ? t
        ? `./${t}`.replace(/\/\.\//g, "/")
        : "."
      : `${n}/${t}`.replace(/([^:]\/)\/+/g, "$1");
  }
  n(2078)(THREE);
  const s = (function () {
    let e,
      t,
      n,
      i,
      s,
      l,
      c = new THREE.Raycaster(),
      u = { w: window.innerWidth, h: window.innerHeight };
    function d(e) {
      if (null == e || "" === e) return !1;
      const t = e.charCodeAt(0);
      return (t >= 13312 && t <= 40959) || (t >= 63744 && t <= 64255);
    }
    function h(e) {
      return null != e && "" !== e && /[a-zA-Z0-9]/.test(e);
    }
    function f() {
      ((this.TWEEN = r.A),
        (this.selectObj = []),
        (this.objects = []),
        (this.optionObj = []),
        (this.multi = !1),
        (this.muscleLayerArr = [[], [], [], [], [], [], [], []]),
        (this.allMuscle = []),
        (this.muscleLayerIndex = 8),
        (this.english = !1),
        (this.selectColor = new THREE.Color(4164159)),
        (this.resetColor = new THREE.Color(0)),
        (this.selectColor1 = new THREE.Color(4164159)),
        (this.selectColor2 = new THREE.Color(52736)),
        (this.mouseoverColor = new THREE.Color(9292941)),
        (this.mouseoverColor2 = new THREE.Color(2414628)),
        (this.mouseoverColor3 = new THREE.Color(16724972)),
        (this.themeAccentColor = new THREE.Color(16768256)),
        (this.themeAccentHoverColor = new THREE.Color(16773017)),
        (this.acupointSpriteBaseColor = new THREE.Color(16777215)),
        (this.mouse = new THREE.Vector2()),
        (this.bodyModelKey = "man"),
        (this.groupLinkToMesh = {}),
        (this.preOpt = null),
        (this.mmObj = null),
        (this.moreLight = !1),
        (this.drag = !1),
        (this.dragObjArr = []),
        (this.showXian = !1),
        (this.heng = !1),
        (this.isolate = !1),
        (this.preIsolateData = {}),
        (this.doubleClick = !1),
        (this.doubleClickTimer = null),
        (this.skinobjArr = []),
        (this.skinDisplayMode = "full"),
        (this.underwearUserVisible = !0),
        (this.changeControlTotTrackball = !1),
        (this.zoomIn = !1),
        (this.distanceIntensity = 6),
        (this.allXuewei = []),
        (this.allSprite = []),
        (this.showRuler = !1),
        (this.rulerMeshArr = []),
        (this.is1K = !1),
        (this.neiMaterialObj = []),
        (this.waiMaterialObj = []),
        (this.sliceUndoBaseStepCount = 0));
    }
    function p() {
      let t = window.innerWidth,
        r = window.innerHeight;
      if (l && l.getBoundingClientRect) {
        const e = l.getBoundingClientRect(),
          n = Math.round(e.width),
          o = Math.round(e.height);
        n >= 1 && o >= 1 && ((t = n), (r = o));
      }
      ((t = Math.max(1, t)),
        (r = Math.max(1, r)),
        (u = { h: r, w: t }),
        (e.aspect = t / r),
        e.updateProjectionMatrix(),
        s.setSize(t, r),
        y.changeControlToTrackball && n.handleResize(),
        y.render());
    }
    function m(e) {
      return e
        ? "Sprite" === e.type && "SPRITE" === e.role && e.xueWeiParent
          ? e.xueWeiParent.dataParent || null
          : e.dataParent || null
        : null;
    }
    function g(e) {
      return (
        !!e.visible &&
        !1 !== e.raycastPickable &&
        !!(function (e) {
          let t = e;
          for (; t; ) {
            if (!1 === t.visible) return !1;
            t = t.parent;
          }
          return !0;
        })(e) &&
        !!(function (e) {
          const t = "Sprite" === e.type && e.xueWeiParent ? e.xueWeiParent : e,
            n = t && t.dataParent;
          if (!n || !n.treeObj) return !0;
          if (y.getGroupLazyLoadState) {
            const e = y.getGroupLazyLoadState(n);
            if ("not_loaded" === e || "loading" === e) return !0;
          }
          return (
            !(
              !n.treeObj.nodeElement || !1 !== n.treeObj.nodeElement.isConnected
            ) || !1 !== n.treeObj.checked
          );
        })(e) &&
        (("Mesh" !== e.type && "SkinnedMesh" !== e.type) || !(0, o.aW)(e)) &&
        !(
          "Mesh" === e.type &&
          y.isSkinRaycastRegulatedMesh &&
          y.isSkinRaycastRegulatedMesh(e) &&
          (!y.isSkinMeshRaycastSolid || !y.isSkinMeshRaycastSolid(e))
        ) &&
        ("Sprite" !== e.type || !(!e.xueWeiParent || !e.xueWeiParent.visible))
      );
    }
    function x(e) {
      ((this.app = e),
        (this.points = []),
        (this.group = null),
        (this.dotTexture = null),
        (this.raycaster = new THREE.Raycaster()),
        (this.mouse = new THREE.Vector2()),
        (this.enabled = !1),
        (this.hoverId = null),
        (this.activeId = null),
        (this.storageKey = "tcm_pain_points_source_v1"),
        (this.ui = null));
    }
    ((x.prototype.load = function () {
      try {
        const e = localStorage.getItem(this.storageKey);
        this.points = e ? JSON.parse(e) : [];
      } catch (e) {
        this.points = [];
      }
      Array.isArray(this.points) || (this.points = []);
    }),
      (x.prototype.save = function () {
        localStorage.setItem(this.storageKey, JSON.stringify(this.points));
      }),
      (x.prototype.ensureGroup = function () {
        this.group ||
          ((this.group = new THREE.Group()),
          (this.group.name = "__pain_points_source__"),
          this.app.scene.add(this.group));
      }),
      (x.prototype.makeMarker = function (e) {
        this.dotTexture ||
          (function (e) {
            const t = document.createElement("canvas");
            ((t.width = 64), (t.height = 64));
            const n = t.getContext("2d");
            (n.clearRect(0, 0, 64, 64),
              (n.fillStyle = "#ff4d4f"),
              n.beginPath(),
              n.arc(32, 32, 12, 0, Math.PI * 2),
              n.fill(),
              (e.dotTexture = new THREE.CanvasTexture(t)),
              (e.dotTexture.needsUpdate = !0));
          })(this);
        const t = new THREE.SpriteMaterial({
            map: this.dotTexture,
            color: 16777215,
            transparent: !0,
            depthWrite: !1,
            depthTest: !0,
          }),
          n = new THREE.Sprite(t);
        return (
          n.scale.set(0.016, 0.016, 1),
          n.material.opacity = 0.9,
          n.position.copy(e),
          n
        );
      }),
      (x.prototype.applyMarkerState = function () {
        if (!this.group) return;
        for (let e = 0; e < this.group.children.length; e++) {
          const t = this.group.children[e],
            n = t.userData ? t.userData.pointId : null,
            r = n && n === this.activeId,
            o = n && n === this.hoverId;
          t.material &&
            (r
              ? ((t.scale.x = t.scale.y = 0.022),
                (t.material.opacity = 1),
                t.material.color.setHex(16767488))
              : o
                ? ((t.scale.x = t.scale.y = 0.019),
                  (t.material.opacity = 0.98),
                  t.material.color.setHex(16764074))
                : ((t.scale.x = t.scale.y = 0.016),
                  (t.material.opacity = 0.88),
                  t.material.color.setHex(16731471)));
        }
      }),
      (x.prototype.getMarkerHit = function (e) {
        const t = this.app.renderer && this.app.renderer.domElement;
        if (!t || !this.group) return null;
        const n = t.getBoundingClientRect();
        ((this.mouse.x = ((e.clientX - n.left) / n.width) * 2 - 1),
          (this.mouse.y = (-((e.clientY - n.top) / n.height)) * 2 + 1),
          this.raycaster.setFromCamera(this.mouse, this.app.camera));
        const r = this.raycaster.intersectObjects(this.group.children, !1);
        return r && r.length ? r[0] : null;
      }),
      (x.prototype.focusPoint = function (e) {
        const t = this.app.camera,
          n = this.app.controls;
        if (!t || !e) return;
        const r = t.position.clone().sub(e).normalize().multiplyScalar(0.55),
          o = e.clone().add(r);
        if (this.app.TWEEN && this.app.TWEEN.Tween) {
          const i = { x: t.position.x, y: t.position.y, z: t.position.z },
            a = n && n.target ? { x: n.target.x, y: n.target.y, z: n.target.z } : null;
          new this.app.TWEEN.Tween(i)
            .to({ x: o.x, y: o.y, z: o.z }, 420)
            .easing(this.app.TWEEN.Easing.Quadratic.Out)
            .onUpdate(() => {
              (t.position.set(i.x, i.y, i.z), n && n.update && n.update(), this.app.render());
            })
            .start();
          a &&
            new this.app.TWEEN.Tween(a)
              .to({ x: e.x, y: e.y, z: e.z }, 420)
              .easing(this.app.TWEEN.Easing.Quadratic.Out)
              .onUpdate(() => {
                (n.target.set(a.x, a.y, a.z), n.update && n.update(), this.app.render());
              })
              .start();
        } else {
          (t.position.copy(o), n && n.target && n.target.copy(e), n && n.update && n.update(), this.app.render());
        }
      }),
      (x.prototype.redraw = function () {
        if (!this.group) return;
        for (; this.group.children.length; ) {
          const e = this.group.children[0];
          (this.group.remove(e), e.material && e.material.dispose && e.material.dispose());
        }
        for (let e = 0; e < this.points.length; e++) {
          const t = this.points[e];
          const n = this.makeMarker(new THREE.Vector3(t.x, t.y, t.z));
          ((n.userData.pointId = t.id), this.group.add(n));
        }
        this.applyMarkerState();
        this.renderList();
      }),
      (x.prototype.renderList = function () {
        if (!this.ui) return;
        const e = this.ui.querySelector(".pain-recorder-list");
        if (!e) return;
        (e.innerHTML = this.points.length
          ? this.points
              .map((e, t) => {
                const n = e.id === this.activeId ? " active" : "",
                  r = e.note ? ` · ${e.note}` : "";
                return `<li class="pain-item${n}" data-id="${e.id}"><span class="pain-dot"></span><span class="pain-text">#${t + 1} ${e.ts || ""}${r}</span></li>`;
              })
              .join("")
          : '<li class="pain-empty">暂无记录</li>');
        const t = e.querySelectorAll(".pain-item[data-id]");
        t.forEach((e) => {
          e.addEventListener("click", () => {
            const t = e.getAttribute("data-id");
            if (!t) return;
            this.activeId = t;
            const n = this.points.find((e) => e.id === t);
            n &&
              (this.focusPoint(new THREE.Vector3(n.x, n.y, n.z)),
              this.ui &&
                this.ui.querySelector(".pain-note-input") &&
                (this.ui.querySelector(".pain-note-input").value = n.note || ""));
            (this.applyMarkerState(), this.renderList(), this.app.render());
          });
        });
      }),
      (x.prototype.removeActive = function () {
        if (!this.activeId) return;
        const e = this.activeId;
        ((this.points = this.points.filter((t) => t.id !== e)),
          (this.activeId = null),
          (this.hoverId = null),
          this.save(),
          this.redraw(),
          this.app.render());
      }),
      (x.prototype.saveActiveNote = function () {
        if (!this.activeId || !this.ui) return;
        const e = this.ui.querySelector(".pain-note-input");
        if (!e) return;
        const t = this.points.find((e) => e.id === this.activeId);
        t && ((t.note = (e.value || "").trim()), this.save(), this.renderList());
      }),
      (x.prototype.bind = function () {
        const e = this.app.renderer && this.app.renderer.domElement;
        e &&
          !e.__painSourceBound &&
          (e.addEventListener(
            "click",
            (t) => {
              if (!this.enabled) return;
              const n = e.getBoundingClientRect();
              ((this.mouse.x = ((t.clientX - n.left) / n.width) * 2 - 1),
                (this.mouse.y = (-((t.clientY - n.top) / n.height)) * 2 + 1),
                this.raycaster.setFromCamera(this.mouse, this.app.camera));
              const o = this.getMarkerHit(t);
              if (o && o.object && o.object.userData && o.object.userData.pointId) {
                const e = o.object.userData.pointId;
                this.activeId = e;
                this.applyMarkerState();
                const t = this.points.find((t) => t.id === e);
                t && this.focusPoint(new THREE.Vector3(t.x, t.y, t.z));
                return;
              }
              const r = [];
              this.app.scene.traverse((e) => {
                e && e.isMesh && r.push(e);
              });
              const i = this.raycaster.intersectObjects(r, !1),
                a = i.length ? i[0].point : this.raycaster.ray.at(1.2, new THREE.Vector3()),
                s = `pain_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
              (this.points.push({
                id: s,
                x: a.x,
                y: a.y,
                z: a.z,
                ts: new Date().toLocaleString(),
                note: "",
              }),
                (this.activeId = s),
                this.save(),
                this.redraw());
              this.ui &&
                this.ui.querySelector(".pain-note-input") &&
                (this.ui.querySelector(".pain-note-input").value = "");
            },
            !0,
          ),
          e.addEventListener(
            "mousemove",
            (e) => {
              const t = this.getMarkerHit(e),
                n =
                  t && t.object && t.object.userData ? t.object.userData.pointId || null : null;
              n !== this.hoverId && ((this.hoverId = n), this.applyMarkerState(), this.app.render());
            },
            !0,
          ),
          (e.__painSourceBound = !0));
      }),
      (x.prototype.mountUI = function () {
        if (document.getElementById("pain-recorder-source")) return;
        const e = document.createElement("div");
        ((e.id = "pain-recorder-source"),
          (e.style.cssText =
            "position:fixed;left:16px;bottom:16px;z-index:999999;background:linear-gradient(180deg,rgba(22,22,26,.92),rgba(12,12,15,.92));border:1px solid rgba(255,255,255,.14);border-radius:12px;color:#fff;padding:10px 10px 8px;width:276px;font:12px 'PingFang SC','Microsoft YaHei',Arial,sans-serif;box-shadow:0 8px 26px rgba(0,0,0,.35);backdrop-filter:blur(6px);"),
          (e.innerHTML =
            '<style>#pain-recorder-source .pain-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}#pain-recorder-source .pain-title{font-size:14px;font-weight:700;letter-spacing:.2px}#pain-recorder-source .pain-sub{font-size:11px;color:#9ca3af;margin-top:2px}#pain-recorder-source .pain-actions{display:flex;gap:8px;margin:8px 0 6px;flex-wrap:wrap}#pain-recorder-source button{height:28px;padding:0 10px;border-radius:8px;border:1px solid rgba(255,255,255,.2);background:#27272f;color:#fff;cursor:pointer}#pain-recorder-source .pain-toggle{background:#7f1d1d;border-color:#ef4444}#pain-recorder-source .pain-clear{background:#23232a}#pain-recorder-source .pain-del{background:#3a1f1f;border-color:#f87171}#pain-recorder-source .pain-note-wrap{display:flex;gap:8px;margin-bottom:8px}#pain-recorder-source .pain-note-input{flex:1;height:28px;border-radius:8px;border:1px solid rgba(255,255,255,.2);background:#17171d;color:#fff;padding:0 8px;outline:none}#pain-recorder-source .pain-recorder-list{max-height:126px;overflow:auto;padding:0;margin:0;list-style:none}#pain-recorder-source .pain-item{display:flex;align-items:center;gap:8px;padding:5px 6px;border-radius:8px;color:#e5e7eb;font-size:12px;cursor:pointer}#pain-recorder-source .pain-item:hover{background:rgba(255,255,255,.08)}#pain-recorder-source .pain-item.active{background:rgba(248,113,113,.18);outline:1px solid rgba(248,113,113,.45)}#pain-recorder-source .pain-dot{width:7px;height:7px;border-radius:50%;background:#ff6e83;}#pain-recorder-source .pain-text{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}#pain-recorder-source .pain-empty{color:#9ca3af;padding:6px}</style><div class=\"pain-head\"><div><div class=\"pain-title\">痛点记录(源码)</div><div class=\"pain-sub\">点击模型记录圆形粒子点</div></div><button class=\"pain-toggle\">开始</button></div><div class=\"pain-actions\"><button class=\"pain-clear\">清空</button><button class=\"pain-del\">删除选中</button></div><div class=\"pain-note-wrap\"><input class=\"pain-note-input\" placeholder=\"选中痛点后填写备注\" /><button class=\"pain-note-save\">保存</button></div><ul class=\"pain-recorder-list\"></ul>'));
        const t = e.querySelector(".pain-toggle"),
          n = e.querySelector(".pain-clear"),
          r = e.querySelector(".pain-del"),
          o = e.querySelector(".pain-note-save");
        (t &&
          t.addEventListener("click", () => {
            ((this.enabled = !this.enabled),
              (t.textContent = this.enabled ? "停止" : "开始"),
              this.enabled ||
                ((this.activeId = null), (this.hoverId = null), this.applyMarkerState(), this.app.render()));
          }),
          n &&
            n.addEventListener("click", () => {
              ((this.points = []), this.save(), this.redraw());
            }),
          r &&
            r.addEventListener("click", () => {
              this.removeActive();
            }),
          o &&
            o.addEventListener("click", () => {
              this.saveActiveNote();
            }),
          document.body.appendChild(e),
          (this.ui = e),
          this.renderList());
      }),
      (x.prototype.start = function () {
        (this.load(), this.ensureGroup(), this.mountUI(), this.bind(), this.redraw());
      }));
    Object.assign(f.prototype, {
      opt: {
        opacity: [],
        hide: [],
        opaque: [],
        show: [],
        step: [],
        select: [],
      },
      flag: { slice: !1, animate: !0 },
      initPainPointRecorder: function () {
        this._painRecorder || ((this._painRecorder = new x(this)), this._painRecorder.start());
      },
      getAcupointSpriteThemeHex: function (e) {
        return e ? 16777215 : 0;
      },
      setAcupointSpriteTheme: function (e) {
        const t = this.getAcupointSpriteThemeHex(e);
        if (
          (this.acupointSpriteBaseColor.setHex(t),
          !this.allSprite || !this.allSprite.length)
        )
          return;
        const n =
          this.selectObj && this.selectObj.length ? this.selectObj[0] : null;
        for (let r = 0; r < this.allSprite.length; r++) {
          const e = this.allSprite[r];
          if (!e || !e.material || !e.material.color) continue;
          const t = e.xueWeiParent && e.xueWeiParent.dataParent;
          t && n === t
            ? e.material.color.copy(this.themeAccentColor)
            : e.material.color.copy(this.acupointSpriteBaseColor);
        }
        this.render();
      },
      init: function (o) {
        ((l = o),
          (this.contain = l),
          (e = new THREE.PerspectiveCamera(
            50,
            window.innerWidth / window.innerHeight,
            0.05,
            10,
          )),
          (i = new THREE.Scene()),
          i.updateMatrixWorld(!0),
          i.add(e),
          i.add(new THREE.AmbientLight(16777215, 0.2)),
          (this.dirLight = new THREE.PointLight(16777215, 0.1)),
          e.add(this.dirLight),
          (this.directionalLight = new THREE.PointLight(13228031, 0.3)),
          this.directionalLight.position.set(3, 2, 1),
          (this.directionalLight2 = new THREE.PointLight(16774635, 0.8)),
          this.directionalLight2.position.set(-3, 2, 1),
          e.add(this.directionalLight),
          e.add(this.directionalLight2),
          (s = new THREE.WebGLRenderer({
            antialias: !0,
            alpha: !0,
            preserveDrawingBuffer: !0,
          })),
          s.setPixelRatio(window.devicePixelRatio),
          s.setSize(window.innerWidth, window.innerHeight),
          (s.outputEncoding = THREE.sRGBEncoding),
          (s.shadowMap.enabled = !0),
          (s.localClippingEnabled = !0),
          (t = new THREE.OrbitControls(e, s.domElement)),
          (t.rotateSpeed = 1.5),
          (t.dynamicDampingFactor = 0.4),
          (t.screenSpacePanning = !0),
          (t.rotateSpeed = 3),
          (t.zoomSpeed = 1.5),
          (t.staticMoving = !0),
          (t.autoRotate = !1),
          t.addEventListener("change", (e) => {
            y.render();
          }),
          (n = new THREE.TrackballControls(e, s.domElement)),
          (n.rotateSpeed = 5),
          (n.zoomSpeed = 4),
          (n.panSpeed = 2),
          (n.staticMoving = !0),
          (n.enabled = !1),
          n.addEventListener("change", () => {
            y.render();
          }),
          (this.camera = e),
          (this.controls = t),
          (this.renderer = s),
          (this.scene = i),
          (this.raycaster = c),
          l.appendChild(s.domElement),
          this.initPainPointRecorder(),
          window.addEventListener("resize", p, !1),
          window.addEventListener("orientationchange", () => {
            setTimeout(() => {
              p();
            }, 400);
          }));
        let d = !1,
          h = 0,
          f = 0;
        function v(e) {
          return e.touches && e.touches[0]
            ? { x: e.touches[0].clientX, y: e.touches[0].clientY }
            : e.changedTouches && e.changedTouches[0]
              ? {
                  x: e.changedTouches[0].clientX,
                  y: e.changedTouches[0].clientY,
                }
              : { x: e.clientX, y: e.clientY };
        }
        function _(e) {
          if (y.draw3D) return;
          y.clearDesktopSceneHoverPick && y.clearDesktopSceneHoverPick();
          const t = v(e);
          ((d = !0), (h = t.x), (f = t.y));
        }
        function b(e) {
          if (!d) return;
          const t = v(e),
            n = t.x - h,
            r = t.y - f;
          n * n + r * r >= 900 && (d = !1);
        }
        function w(t) {
          if (!d) return;
          if (y.draw3D) return;
          (!(function (t) {
            if (y.draw3D) return;
            !t.clientX && t.changedTouches && (t = t.changedTouches[0]);
            let n = null;
            const { clientX: o, clientY: i } = t,
              a =
                l && l.getBoundingClientRect
                  ? l.getBoundingClientRect()
                  : { left: 0, top: 0, width: u.w, height: u.h };
            ((y.mouse.x = ((o - a.left) / a.width) * 2 - 1),
              (y.mouse.y = (-(i - a.top) / a.height) * 2 + 1),
              c.setFromCamera(y.mouse, e));
            const s = y.objects.filter((e) => g(e));
            let d = c.intersectObjects(s);
            if (d.length > 0) {
              ((n = d[0].object), console.log(n));
              const e = d[0].point;
              if (
                "Mesh" === n.type &&
                y.isSkinMeshRaycastSolid &&
                y.isSkinMeshRaycastSolid(n)
              )
                return void y.setEditing(!1);
              if (
                "SPRITE" === n.role &&
                n.xueWeiParent &&
                n.xueWeiParent.visible
              )
                return void y.clickOption(n.xueWeiParent.dataParent);
              if (y.patientToolType) {
                if ("pain" === y.patientToolType)
                  return void y.painEffect(e, n);
                if ("tumour" === y.patientToolType) {
                  if (y.zhongliuObj) {
                    let t = y.zhongliuObj.clone();
                    (t.position.copy(e),
                      (t.visible = !0),
                      (t.role = "patientToolObj"),
                      y.objects.push(t),
                      (t.parentObj = n),
                      n.children.push(t),
                      y.beginEditingpatientTool(t));
                  }
                  return;
                }
                if ("jiantou" === y.patientToolType) {
                  if (y.jiantouObj) {
                    const t = y.jiantouObj.clone();
                    return (
                      t.position.copy(e),
                      t.quaternion.copy(y.camera.quaternion),
                      (t.visible = !0),
                      (t.role = "patientToolObj"),
                      y.scene.add(t),
                      y.objects.push(t),
                      (t.parentObj = n),
                      n.children.push(t),
                      void y.beginEditingpatientTool(t, { x: o, y: i })
                    );
                  }
                  return;
                }
                if ("zhentong" === y.patientToolType) {
                  if (y.zhentongObj) {
                    const t = y.zhentongObj.clone();
                    return (
                      t.position.copy(e),
                      t.quaternion.copy(y.camera.quaternion),
                      (t.visible = !0),
                      (t.role = "patientToolObj"),
                      y.scene.add(t),
                      y.objects.push(t),
                      (t.parentObj = n),
                      n.children.push(t),
                      void y.beginEditingpatientTool(t, { x: o, y: i })
                    );
                  }
                  return;
                }
              }
              if (y.addLabel) {
                const t =
                  n.dataParent || (n.xueWeiParent && n.xueWeiParent.dataParent);
                return void y.beginAddLabel(e, t && t.cnName, { x: o, y: i });
              }
              if ("patientToolObj" === n.role)
                return void (n.visible && y.beginEditingpatientTool(n));
              if ((y.setEditing(!1), "Sprite" === n.type)) {
                if (y.doubleClick && !y.drag) {
                  if (y.selectObj.length) return void y.lookAtSelectObj();
                } else y.handDoubleClick();
                (y.clickOption(n.xueWeiParent.dataParent), y.doubleClick);
              } else if (n && n.dataParent) {
                if (!y.multi)
                  if (y.doubleClick) {
                    if (y.drag) {
                      let e = n._position;
                      return (
                        new r.A.Tween(n.position)
                          .to({ x: e.x, y: e.y, z: e.z }, 600)
                          .easing(r.A.Easing.Exponential.Out)
                          .start()
                          .onUpdate(() => {
                            y.render();
                          }),
                        y.dragObjArr.remove(n),
                        void console.log(y.dragObjArr)
                      );
                    }
                    var h, f, p, m;
                    if (y.selectObj.length && y.selectObj[0] === n.dataParent)
                      return (
                        (y.zoomIn = !1),
                        null === (h = y.optionBar) ||
                          void 0 === h ||
                          null === (f = h.setState) ||
                          void 0 === f ||
                          f.call(h, { zoomIn: !1 }),
                        y.lookAtSelectObj(),
                        (y.zoomIn = !1),
                        void (
                          null === (p = y.optionBar) ||
                          void 0 === p ||
                          null === (m = p.setState) ||
                          void 0 === m ||
                          m.call(p, { zoomIn: !1 })
                        )
                      );
                  } else y.handDoubleClick();
                (y.clickOption(n.dataParent, !1, n), y.doubleClick);
              }
            } else {
              var v, _, b, w;
              if ((y.setEditing(!1), y.isolate))
                null === (v = y.optionBar) ||
                  void 0 === v ||
                  null === (_ = v.setState) ||
                  void 0 === _ ||
                  _.call(v, { fullScreen: !0 });
              else
                null === (b = y.optionBar) ||
                  void 0 === b ||
                  null === (w = b.setState) ||
                  void 0 === w ||
                  w.call(b, { show: !1, cnName: null, sysNote: null });
              (y.multi || y.resetSelectObjColor(),
                y.mmObj && y.changeMMDye(!1),
                (y.mmObj = null),
                y.render());
            }
          })(t.changedTouches && t.changedTouches[0] ? t.changedTouches[0] : t),
            (d = !1));
        }
        const x = s.domElement;
        (x.addEventListener("pointerdown", _, !1),
          x.addEventListener("pointermove", b, !1),
          x.addEventListener("pointerup", w, !1),
          x.addEventListener(
            "pointercancel",
            () => {
              d = !1;
            },
            !1,
          ),
          "PointerEvent" in window ||
            (x.addEventListener("touchstart", _, { passive: !0 }),
            x.addEventListener("touchmove", b, { passive: !0 }),
            x.addEventListener("touchend", w, !1)),
          (y.desktopSceneHover =
            "undefined" !== typeof window &&
            window.matchMedia &&
            window.matchMedia("(pointer: fine)").matches &&
            window.matchMedia("(hover: hover)").matches));
        let S = null,
          L = null;
        function R(t, n) {
          ((L = { clientX: t, clientY: n }),
            null == S &&
              (S = requestAnimationFrame(() => {
                S = null;
                const t = L;
                ((L = null),
                  t &&
                    (function (t, n) {
                      if (!y.desktopSceneHover || !y.updateDesktopHoverTip)
                        return;
                      if (y.ifMousemove) return;
                      if (y.draw3D || y.drag) return;
                      const r =
                        l && l.getBoundingClientRect
                          ? l.getBoundingClientRect()
                          : { left: 0, top: 0, width: u.w, height: u.h };
                      ((y.mouse.x = ((t - r.left) / r.width) * 2 - 1),
                        (y.mouse.y = (-(n - r.top) / r.height) * 2 + 1),
                        c.setFromCamera(y.mouse, e));
                      const o = y.objects.filter((e) => g(e)),
                        i = c.intersectObjects(o, !1);
                      let a = null;
                      if (
                        (i.length > 0 && (a = i[0].object),
                        a &&
                          "Mesh" === a.type &&
                          y.isSkinMeshRaycastSolid &&
                          y.isSkinMeshRaycastSolid(a))
                      )
                        return (
                          y.mmObj &&
                            m(y.mmObj) !== y.selectObj[0] &&
                            y.changeMMDye(!1),
                          y.updateDesktopHoverTip(null),
                          (y.mmObj = null),
                          void (y.render && y.render())
                        );
                      const s = a ? m(a) : null,
                        d =
                          y.selectObj && y.selectObj.length
                            ? y.selectObj[0]
                            : null;
                      if (a && s) {
                        const e = s.cnName,
                          r = y.english
                            ? s.enName || e || ""
                            : (y.GetChinese ? y.GetChinese(e) : e) || e || "";
                        if (
                          (y.updateDesktopHoverTip({
                            text: r,
                            clientX: t,
                            clientY: n,
                          }),
                          a === y.mmObj)
                        )
                          return;
                        if (s === d) return;
                        (y.mmObj && (y.changeMMDye(!1), (y.mmObj = null)),
                          (y.mmObj = a),
                          y.changeMMDye(!0));
                      } else
                        (y.mmObj && m(y.mmObj) !== d && y.changeMMDye(!1),
                          y.updateDesktopHoverTip(null),
                          (y.mmObj = null));
                      y.render && y.render();
                    })(t.clientX, t.clientY));
              })));
        }
        ((y.clearDesktopSceneHoverPick = function () {
          if (
            (y.updateDesktopHoverTip && y.updateDesktopHoverTip(null),
            !y.desktopSceneHover)
          )
            return;
          const e = y.selectObj && y.selectObj.length ? y.selectObj[0] : null;
          (y.mmObj && m(y.mmObj) !== e && (y.changeMMDye(!1), (y.mmObj = null)),
            y.render && y.render());
        }),
          y.desktopSceneHover &&
            (x.addEventListener(
              "pointermove",
              function (e) {
                y.desktopSceneHover &&
                  ((e.pointerType && "mouse" !== e.pointerType) ||
                    (0 === e.buttons &&
                      (y.draw3D ||
                        y.drag ||
                        y.ifMousemove ||
                        R(e.clientX, e.clientY))));
              },
              !1,
            ),
            x.addEventListener(
              "pointerleave",
              function () {
                if (!y.desktopSceneHover) return;
                ((L = null),
                  null != S && (cancelAnimationFrame(S), (S = null)));
                const e =
                  y.selectObj && y.selectObj.length ? y.selectObj[0] : null;
                (y.mmObj && m(y.mmObj) !== e && y.changeMMDye(!1),
                  y.updateDesktopHoverTip && y.updateDesktopHoverTip(null),
                  (y.mmObj = null),
                  y.render && y.render());
              },
              !1,
            )),
          (y.effect = new THREE.OutlineEffect(s)));
        const E = 1.5,
          C = (e, t) => {
            const n = document.createElement("canvas");
            n.width = n.height = 64;
            const r = n.getContext("2d");
            ((r.fillStyle = e), r.fillRect(0, 0, 64, 64), (r.fillStyle = t));
            for (let o = 0; o < 8; o++) r.fillRect(8 * o, 0, 4, 64);
            return n.toDataURL("image/png");
          },
          k = new THREE.TextureLoader();
        ((y.map = k.load(
          a("img/BW.png"),
          (e) => {
            ((e.wrapS = e.wrapT = THREE.RepeatWrapping), e.repeat.set(E, E));
          },
          void 0,
          () => {
            ((y.map = k.load(C("#0066ff", "#ffffff"))),
              (y.map.wrapS = y.map.wrapT = THREE.RepeatWrapping),
              y.map.repeat.set(E, E),
              (y.BWanimateMaterial.map = y.map));
          },
        )),
          (y.map.wrapS = y.map.wrapT = THREE.RepeatWrapping),
          y.map.repeat.set(E, E),
          (y.BWanimateMaterial = new THREE.MeshBasicMaterial({
            transparent: !0,
            map: y.map,
            color: 16777215,
          })),
          (y.map2 = k.load(
            a("img/RW.png"),
            (e) => {
              ((e.wrapS = e.wrapT = THREE.RepeatWrapping), e.repeat.set(E, E));
            },
            void 0,
            () => {
              ((y.map2 = k.load(C("#cc0000", "#ffffff"))),
                (y.map2.wrapS = y.map2.wrapT = THREE.RepeatWrapping),
                y.map2.repeat.set(E, E),
                (y.RWanimateMaterial.map = y.map2));
            },
          )),
          (y.map2.wrapS = y.map2.wrapT = THREE.RepeatWrapping),
          y.map2.repeat.set(E, E),
          (y.RWanimateMaterial = new THREE.MeshBasicMaterial({
            map: y.map2,
            color: 16777215,
          })),
          this.buildAllXmlGroup(i),
          (this.render = function () {
            y.flag.animate ||
              (y.showXian ? y.effect.render(i, e) : s.render(i, e));
          }));
        let M = 0;
        const A = 0.07;
        let P = 0;
        function T() {
          if (
            (requestAnimationFrame(T),
            r.A.update(),
            y.flag.animate && y.map && y.map2 && y.map.image && y.map2.image)
          ) {
            P += 0.007;
            const e = P % 1;
            ((y.BWanimateMaterial.map.offset.y = e),
              (y.RWanimateMaterial.map.offset.y = e));
          }
          if (y.painList && y.painList.length) {
            M += A;
            const e = 0.5 * (0.5 * Math.sin(M) + 0.5) + 0.2;
            y.painList.forEach((t) => {
              t.material.uniforms.opacity.value = e;
            });
          }
          if (
            (y.changeControlToTrackball && n.update(),
            y.showXian ? y.effect.render(i, e) : s.render(i, e),
            y.labelList &&
              y.labelList.length &&
              y.labelList.forEach((e) => {
                const { point: t, label: n } = e;
                if (n) {
                  const { instance: e, dotDiv: r } = n;
                  let o = new THREE.Vector3(t.x, t.y, t.z);
                  o.project(y.camera);
                  let i = Math.round(((o.x + 1) * u.w) / 2),
                    a = Math.round(((1 - o.y) * u.h) / 2);
                  ((r.style.left = i + "px"),
                    (r.style.top = a + "px"),
                    e.repaintEverything());
                }
              }),
            y.labelEditingData)
          ) {
            const { point: e, dotDiv: t } = y.labelEditingData;
            let n = new THREE.Vector3(e.x, e.y, e.z);
            n.project(y.camera);
            let r = Math.round(((n.x + 1) * u.w) / 2),
              o = Math.round(((1 - n.y) * u.h) / 2);
            ((t.style.left = r + "px"), (t.style.top = o + "px"));
          }
        }
        (p(),
          T(),
          (Array.prototype.remove = function (e) {
            var t = this.indexOf(e);
            t > -1 && this.splice(t, 1);
          }));
      },
      handleSwitchControle() {
        ((y.changeControlToTrackball = !y.changeControlToTrackball),
          n ||
            ((n = new THREE.TrackballControls(y.camera, y.renderer.domElement)),
            (n.rotateSpeed = 6),
            (n.zoomSpeed = 4),
            (n.panSpeed = 2),
            (n.staticMoving = !0),
            (n.dynamicDampingFactor = 0.3),
            (n.enabled = !1)),
          console.log(y.changeControlToTrackball),
          y.changeControlToTrackball
            ? ((t.enabled = !1),
              (y.controls = n),
              (n.enabled = !0),
              n.handleResize())
            : ((n.enabled = !1), (y.controls = t), (t.enabled = !0)),
          y.camera.up.set(0, 1, 0),
          y.initCamarePos(!0),
          y.forcePan && (y.controls.forcePan = y.forcePan),
          y.DirectionComponent &&
            "function" === typeof y.DirectionComponent.applyPanModeToControls &&
            y.DirectionComponent.applyPanModeToControls(
              y.controls,
              !!y.forcePan,
            ));
      },
      handDoubleClick() {
        ((this.doubleClick = !0),
          this.doubleClickTimer && clearInterval(this.doubleClickTimer),
          (this.doubleClickTimer = setTimeout(() => {
            this.doubleClick = !1;
          }, 300)));
      },
      getPreIsolateData() {
        ((this.preIsolateData = {}), (this.preIsolateData.selectList = []));
        let e = [],
          t = [];
        for (var n = 0; n < y.optionObj.length; n++)
          (y.optionObj[n].visible && e.push(y.optionObj[n].name),
            1 !== y.optionObj[n].material.opacity &&
              t.push(y.optionObj[n].name));
        for (n = 0; n < y.selectObj.length; n++)
          this.preIsolateData.selectList.push(y.selectObj[n].name);
        ((this.preIsolateData.name = "5"),
          (this.preIsolateData.opacityList = t),
          (this.preIsolateData.objectList = e),
          (this.preIsolateData.position = JSON.stringify(y.camera.position)),
          (this.preIsolateData.target = JSON.stringify(y.controls.target)));
      },
      calcSelectObjSphere(e) {
        let t = e || this.selectObj;
        if (!t || !t.length) return;
        let n = new THREE.Object3D();
        for (let i = 0, a = t.length; i < a; i++) n.children.push(t[i]);
        let r = new THREE.Box3();
        r.expandByObject(n);
        let o = new THREE.Sphere();
        if ((r.getBoundingSphere(o), o.radius <= 0 && 1 === t.length)) {
          const e = t[0];
          if (
            e &&
            ("XW" === e.type || "XW" === e.role) &&
            e.children &&
            e.children.length
          ) {
            const t = e.children.find(
              (e) => e.isMesh && ("xw" === e.role || "XWMESH" === e.role),
            );
            t &&
              (t.updateMatrixWorld(!0),
              o.center.copy(t.getWorldPosition(new THREE.Vector3())),
              (o.radius = 0.15));
          }
        }
        return o;
      },
      lookAtSelectObj(e, t) {
        r.A.removeAll();
        const n = this.calcSelectObjSphere(e);
        if (!n) return;
        let { center: o, radius: i } = n;
        (i <= 0 && (i = 0.15), y.camera.lookAt(o));
        let a,
          s,
          l,
          c = y.camera.position.clone(),
          u = c.distanceTo(o);
        if (t || !y.zoomIn)
          ((a = i * y.distanceIntensity), (s = a / u), (l = c.lerp(o, 1 - s)));
        else {
          a = y.rootObjectRadius;
          let e = new THREE.Vector3(c.x, c.y, c.z);
          ((u = e.distanceTo(y.rootObjCenter)),
            (s = a / u),
            (l = e.lerp(y.rootObjCenter, 1 - s)));
        }
        const d = t ? 800 : 500,
          h = t ? r.A.Easing.Exponential.Out : void 0,
          f = new r.A.Tween(y.controls.target).to(
            { x: o.x, y: o.y, z: o.z },
            d,
          );
        (h && f.easing(h), f.start());
        const p = new r.A.Tween(y.camera.position).to(
          { x: l.x, y: l.y, z: l.z },
          d,
        );
        (h && p.easing(h),
          p.start().onUpdate(() => {
            (y.camera.lookAt(y.controls.target), y.render());
          }));
      },
      calcZoomInState(e) {
        var t, n;
        const r = this.calcSelectObjSphere(e),
          { center: o, radius: i } = r;
        let a = y.camera.position.clone(),
          s = a.distanceTo(o) > i * (1.01 * y.distanceIntensity);
        (s || 0 !== a.y || (s = !0),
          null === (t = y.optionBar) ||
            void 0 === t ||
            null === (n = t.setState) ||
            void 0 === n ||
            n.call(t, { zoomIn: s }),
          (y.zoomIn = s));
      },
      changeMMDye: function (e) {
        if (!y.mmObj) return;
        if ("Sprite" === y.mmObj.type && "SPRITE" === y.mmObj.role) {
          if (e) y.mmObj.material.color.copy(this.themeAccentHoverColor);
          else {
            const e = y.mmObj.xueWeiParent,
              t = e && e.dataParent,
              n = y.selectObj && y.selectObj.length ? y.selectObj[0] : null;
            t && n === t
              ? y.mmObj.material.color.copy(this.themeAccentColor)
              : y.mmObj.material.color.copy(this.acupointSpriteBaseColor);
          }
          return;
        }
        const t = y.neiMaterialObj && y.neiMaterialObj.indexOf(y.mmObj) >= 0,
          n = y.waiMaterialObj && y.waiMaterialObj.indexOf(y.mmObj) >= 0;
        if (y.flag.animate && (t || n)) {
          const e = t ? y.BWanimateMaterial : y.RWanimateMaterial;
          return void (y.mmObj.material =
            y.mmObj.selected && y.mmObj.animateSelectMaterial
              ? y.mmObj.animateSelectMaterial
              : e);
        }
        if (y.showXian) {
          (r = y.mmObj.material).color = e
            ? this.mouseoverColor
            : r.originalColor;
        } else {
          var r = y.mmObj.originalMaterial;
          if (e) {
            let e;
            (y.mmObj.activeMaterial
              ? (e = y.mmObj.activeMaterial)
              : ((e = r.clone()), (y.mmObj.activeMaterial = e)),
              (y.mmObj.material = e),
              !y.mmObj.dataParent.role ||
              ("sj" !== y.mmObj.dataParent.role &&
                "jm" !== y.mmObj.dataParent.role &&
                "dm" !== y.mmObj.dataParent.role &&
                "zwsj" !== y.mmObj.dataParent.role)
                ? !y.mmObj.dataParent.role ||
                  ("gg" !== y.mmObj.dataParent.role &&
                    "glj" !== y.mmObj.dataParent.role)
                  ? "lb" === y.mmObj.dataParent.role
                    ? ((e.color = this.mouseoverColor3),
                      (e.emissiveIntensity = 0.7))
                    : ((e.color = this.mouseoverColor),
                      (e.emissiveIntensity = 0.4))
                  : (e.color = this.mouseoverColor)
                : ((e.color = this.mouseoverColor2),
                  (e.emissiveIntensity = 0.4)));
          } else
            y.mmObj.activeMaterial &&
              ((y.mmObj.activeMaterial.color = r.originalColor),
              (y.mmObj.activeMaterial.emissiveIntensity = 0));
        }
      },
      parseURL: function (e) {
        var t = document.createElement("a");
        return (
          (t.href = e),
          {
            params: (function () {
              for (
                var e,
                  n = {},
                  r = t.search.replace(/^\?/, "").split("&"),
                  o = r.length,
                  i = 0;
                i < o;
                i++
              )
                r[i] && (n[(e = r[i].split("="))[0]] = e[1]);
              return n;
            })(),
          }
        );
      },
      buildAllXmlGroup: function (e) {
        let t = [];
        const n = new THREE.Group();
        ((n.objName = objlist.root.o),
          (n.cnName = objlist.root.c),
          (n.root = !0));
        let r = objlist.root.ch;
        function o(e) {
          if (!e) return null;
          const t = y.adjustName(e);
          for (let n = 0; n < partList.list.length; n++)
            if (y.adjustName(partList.list[n].objName) === t)
              return partList.list[n].role;
          return null;
        }
        for (let i = 0; i < r.length; i++) {
          const e = o(r[i].o);
          null != e && (r[i].role = e);
        }
        (!(function e(n, r) {
          if (n.length)
            n.forEach(function (n) {
              var o;
              let i = new THREE.Group(),
                a = y.adjustName(n.o);
              ((i.opacity = !1),
                (i.name = "Group_" + a),
                (i.objName = n.o),
                (i.cnName = n.c),
                (i.enName = n.e),
                n.s && (i.sysNote = n.s),
                n.l && (i.layer = n.l),
                n.xw && (i.xw = n.xw),
                n.type && (i.type = n.type),
                n.ids && (i.ids = n.ids),
                n.JB && (i.JB = n.JB),
                n.LM && (i.LM = n.LM),
                n.JJ && (i.JJ = n.JJ),
                n.PB && (i.PB = n.PB),
                n.ZJ && (i.ZJ = n.ZJ));
              const s = null !== (o = n.role) && void 0 !== o ? o : r.role;
              if (
                (null != s && (i.role = s),
                (y.groupLinkToMesh[a] = i),
                r.add(i),
                n.ch)
              )
                return e(n.ch, i);
              t.push(i);
            });
          else {
            var o;
            let i = new THREE.Group(),
              a = y.adjustName(n.o);
            ((i.opacity = !1),
              (i.name = "Group_" + a),
              (i.objName = n.o),
              (i.cnName = n.c),
              (i.enName = n.e),
              n.s && (i.sysNote = n.s));
            const s = null !== (o = n.role) && void 0 !== o ? o : r.role;
            if (
              (null != s && (i.role = s),
              n.xw && (i.xw = n.xw),
              n.type && (i.type = n.type),
              n.ids && (i.ids = n.ids),
              n.JB && (i.JB = n.JB),
              n.LM && (i.LM = n.LM),
              n.JJ && (i.JJ = n.JJ),
              n.PB && (i.PB = n.PB),
              n.ZJ && (i.ZJ = n.ZJ),
              n.l && (i.layer = n.l),
              y.groupLinkToMesh[a] && console.log(n.o),
              (y.groupLinkToMesh[a] = i),
              r.add(i),
              n.ch)
            )
              return e(n.ch, i);
            t.push(i);
          }
        })(r, n),
          e.add(n));
      },
      mousePick: function (t, n) {
        if (y.draw3D) return;
        if (y.drag || y.draw3D) return;
        let r = null;
        ((y.mouse.x = (t.x / u.w) * 2 - 1),
          (y.mouse.y = (-t.y / u.h) * 2 + 1),
          c.setFromCamera(y.mouse, e));
        let o = c.intersectObjects(
          y.objects.filter((e) => g(e)),
          !1,
        );
        if (o.length > 0) {
          if (
            ((r = o[0].object),
            "Mesh" === r.type &&
              y.isSkinMeshRaycastSolid &&
              y.isSkinMeshRaycastSolid(r))
          )
            return (
              y.mmObj && m(y.mmObj) != y.selectObj[0] && y.changeMMDye(!1),
              n.setState({ mousemoveName: null }),
              (y.mmObj = null),
              void y.render()
            );
          const e = m(r);
          if (e) {
            let t = e.cnName;
            n.setState({ mousemoveName: y.english ? e.enName : t });
          }
          if (r === y.mmObj) return;
          if (e === y.selectObj[0]) return;
          (y.mmObj && (y.changeMMDye(!1), (y.mmObj = null)),
            (y.mmObj = r),
            y.changeMMDye(!0));
        } else
          (y.mmObj && m(y.mmObj) != y.selectObj[0] && y.changeMMDye(!1),
            n.setState({ mousemoveName: null }),
            (y.mmObj = null));
        y.render();
      },
      mouseSelectClick: function () {
        const e = y.mmObj ? m(y.mmObj) : null;
        y.mmObj && e && y.clickOption(e);
      },
      changeAnimateMaterial: function (e) {
        if (!y.BWanimateMaterial || !y.RWanimateMaterial) return;
        const t = y.selectObj && y.selectObj.length ? [...y.selectObj] : [],
          n = [];
        t.forEach(function (e) {
          e && e.length
            ? e.forEach(function (e) {
                e && n.push(e);
              })
            : e && n.push(e);
        });
        const r = this,
          o = () => {
            t.length &&
              y.changeSelectDye &&
              t.forEach(function (e) {
                e && e.length
                  ? e.forEach(function (e) {
                      e && y.changeSelectDye(e, !0);
                    })
                  : e && y.changeSelectDye(e, !0);
              });
          },
          i = () => {
            y._acupointLinkedMeridianRoot &&
              y.applyMeridianSubtreeThickness &&
              y.applyMeridianSubtreeThickness(
                y._acupointLinkedMeridianRoot,
                !0,
              );
          },
          a = () => {
            const t = e && n.length && y.shouldThickenNeiWaiMeshForSelection;
            for (let o = 0; o < r.neiMaterialObj.length; o++) {
              const i = r.neiMaterialObj[o];
              if (
                (y.setMeridianThickness && y.setMeridianThickness(i, !1),
                (i.material = e ? y.BWanimateMaterial : i.originalMaterial),
                (i.selected = !1),
                t &&
                  i.dataParent &&
                  y.shouldThickenNeiWaiMeshForSelection(i, n))
              ) {
                const e = y.BWanimateMaterial;
                let t = i.animateSelectMaterial;
                (t || ((t = e.clone()), (i.animateSelectMaterial = t)),
                  t.color.setHex(16777215),
                  (i.material = t),
                  (i.selected = !0),
                  y.setMeridianThickness && y.setMeridianThickness(i, !0),
                  y.setAcupointModelScale && y.setAcupointModelScale(i, !0));
              }
            }
            for (let o = 0; o < r.waiMaterialObj.length; o++) {
              const i = r.waiMaterialObj[o];
              if (
                (y.setMeridianThickness && y.setMeridianThickness(i, !1),
                (i.material = e ? y.RWanimateMaterial : i.originalMaterial),
                (i.selected = !1),
                t &&
                  i.dataParent &&
                  y.shouldThickenNeiWaiMeshForSelection(i, n))
              ) {
                const e = y.RWanimateMaterial;
                let t = i.animateSelectMaterial;
                (t || ((t = e.clone()), (i.animateSelectMaterial = t)),
                  t.color.setHex(16777215),
                  (i.material = t),
                  (i.selected = !0),
                  y.setMeridianThickness && y.setMeridianThickness(i, !0),
                  y.setAcupointModelScale && y.setAcupointModelScale(i, !0));
              }
            }
          };
        (a(),
          e &&
            t.length &&
            (o(),
            requestAnimationFrame(() => {
              (a(), y.render && y.render());
            })),
          !e &&
            t.length &&
            (o(),
            i(),
            requestAnimationFrame(() => {
              (i(), y.render && y.render());
            })),
          y.render && y.render());
      },
      calcPosition: function (e) {
        (e.updateMatrixWorld(!0),
          (e.matrixWorldNeedsUpdate = !0),
          e.geometry.computeBoundingBox());
        var t = e.geometry.boundingBox,
          n = new THREE.Vector3();
        return (
          n.subVectors(t.max, t.min),
          n.multiplyScalar(0.5),
          n.add(t.min),
          n.applyMatrix4(e.matrixWorld),
          n
        );
      },
      resetControl: function (e) {
        (0 === y.camera.position.y && 0 === y.camera.position.z) ||
          (y.controls.target.set(0, 0, 0),
          y.camera.up.set(0, 1, 0),
          new r.A.Tween(y.camera.position)
            .to({ x: 0, y: 0, z: y.rootObjectRadius }, e ? 100 : 800)
            .easing(r.A.Easing.Exponential.Out)
            .onUpdate(() => {
              (y.camera.lookAt(y.controls.target), y.render());
            })
            .start());
      },
      initCamarePos: (e) => {
        const t = new THREE.Box3();
        t.expandByObject(y.rootObject);
        const n = new THREE.Sphere();
        t.getBoundingSphere(n);
        const { center: o, radius: i } = n;
        ((y.rootObjCenter = o),
          "pelvis" === y.urlPart
            ? (y.rootObjectRadius = 2.6 * i)
            : "thorax" === y.urlPart
              ? (y.rootObjectRadius = 4 * i)
              : "man" === y.urlPart || "female" === y.urlPart
                ? (y.rootObjectRadius = 2.2)
                : "spine" === y.urlPart
                  ? (y.rootObjectRadius = 3 * i)
                  : "hand" === y.urlPart
                    ? (y.rootObjectRadius = 2.7 * i)
                    : (y.rootObjectRadius = 3.7 * i));
        let a = e ? 0 : 500;
        (new r.A.Tween(y.controls.target)
          .to({ x: o.x, y: o.y, z: o.z }, a)
          .start(),
          new r.A.Tween(y.camera.position)
            .to({ x: o.x, y: o.y, z: y.rootObjectRadius }, a)
            .start()
            .onUpdate(() => {
              (y.camera.lookAt(y.controls.target), y.render());
            }));
      },
      GetChinese: function (e) {
        if (null == e || "" === e) return e;
        const t = String(e),
          { name: n } = this.splitCnEnFromC(t);
        return void 0 !== n && "" !== n ? n : t;
      },
      splitCnEnFromC: function (e) {
        const t = (null == e ? "" : String(e)).trim();
        if (!t) return { name: "", enName: "" };
        let n = -1;
        for (let r = 1; r < t.length; r++)
          if (d(t[r - 1]) && h(t[r])) {
            n = r;
            break;
          }
        if (-1 === n) {
          let e = !1;
          for (let r = 0; r < t.length; r++)
            if ((d(t[r]) && (e = !0), e && h(t[r]))) {
              n = r;
              break;
            }
        }
        if (-1 === n) {
          return /[\u3400-\u9FFF]/.test(t) || /[\uF900-\uFAFF]/.test(t)
            ? { name: t, enName: "" }
            : { name: "", enName: t };
        }
        return { name: t.slice(0, n).trim(), enName: t.slice(n).trim() };
      },
      GetEnglish: function (e) {
        const t = /[\u4e00-\u9fa5]/g;
        if (null != e && "" != e) {
          let n = 0;
          for (let r = e.length - 1; r >= 0; r--) {
            if (e[r].match(t)) {
              n = r + 1;
              break;
            }
          }
          return n ? e.slice(n, e.length) : e;
        }
      },
      adjustName: function (e) {
        if (!e) return;
        let t = e.replace(/ /g, "_");
        return (
          [":", ",", ";", "(", ")", "'", "-", "]", "[", "+"].forEach(
            function (e, n) {
              ((t = t.split(e)), (t = t.join("_")));
            },
          ),
          t
        );
      },
      getOriginalEvent: function (e) {
        return (
          e && "undefined" !== typeof e.originalEvent && (e = e.originalEvent),
          e
        );
      },
      resetApp: () => {
        for (var e = 0; e < y.optionObj.length; e++)
          ((y.optionObj[e].visible = !1),
            y.optionObj[e].dataParent && y.optionObj[e].dataParent.treeObj
              ? y.optionObj[e].dataParent.treeObj.setState({
                  checked: !1,
                  selected: !1,
                })
              : console.log(y.optionObj[e].name + "\u6ca1\u6709dataParent"));
        ((y.opt = {
          opacity: [],
          hide: [],
          opaque: [],
          show: [],
          step: [],
          select: [],
        }),
          (y.selectObj = []),
          (y.optionObj = []),
          (y.sliceUndoBaseStepCount = 0));
      },
    });
    const y = new f();
    return ((y.windowResize = p), (window.app = y), y);
  })();
};
